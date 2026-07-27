# htsglang Fork Features

An SGLang fork for heterogeneous / mismatched GPUs. This file lists the capabilities the fork
carries beyond upstream SGLang, and whether an equivalent capability exists in upstream SGLang,
upstream vLLM, `llama.cpp` or `ik_llama.cpp`. All entries are work in progress; no entry implies
external review or upstream-mergeable maturity.

## Status legend

| Tier | Meaning |
|---|---|
| `Built` | Merged; covered by the fork's own tests only. |
| `Boot-checked` | Executed on hardware with a real model; coherent output. |
| `Cross-checked` | Validated against an independent reference — another backend, a solo/TP=1 run as oracle, `torch`/`torch.distributed`, or a byte-/token-identity that must hold for structural reasons. The reference is named per row. |

Modifiers: `WIP` — present but not complete. `Exp` — experimental, not production-ready. A
trailing `*` — the capability lives only on an unmerged branch. `red-then-green` marks a row whose
own test fails before its fix and passes after.

Two identities do **not** qualify as a cross-check:

| Claim | Why it does not hold |
|---|---|
| Byte-/token-identity above ~109 prompt tokens on an RTX 3080 under fp8 | `gptq_marlin_gemm`, the only fp8 GEMM sm86 has, is measured run-to-run nondeterministic above that length: 0 of 1200 mismatches through M=109, first mismatch at M=128. The fix is not merged. The RTX 5090 (sm120, a different fp8 GEMM path) is unaffected at any length. |
| Token identity between speculative and non-speculative decoding | The verify round computes k+1 tokens in one forward instead of one per token, so the reduction order differs. With repetition, presence or frequency penalties set, n-1 of n accepted tokens never reach the penalty function. The acceptance decision itself is exact integer equality against the target's argmax and cannot diverge. A valid reference for a speculative arm carries the same speculative configuration. |

## Reference hardware

| Role | Cards and link |
|---|---|
| Main rig | 1x RTX 5090 (sm120, 32 GB) + 2x RTX 3080 (sm86, 20 GB) |
| Second host | 1x RTX 2080 Ti (sm75) + 1x Radeon RX Vega 64 (gfx900, 8.0 GB) |
| Interconnect | No NVLink, no CUDA P2P (GeForce, PHB topology); all cross-GPU traffic host-staged. One 3080 on PCIe Gen4 x4 at ~6.5 GB/s host-staged DMA against ~13-14 GB/s for the other two. Cross-host: 40G RoCE for data, 1 GbE for the control plane. The main rig's NIC negotiates PCIe 3.0 x4 against an x16 LnkCap, so cross-host transfers cap at 3.43 GB/s: `ib_write_bw` at 1 MiB returns 3270 MiB/s over the 40G and the 100G port alike. `ib_write_lat` at 8 B is 1.47 us over 40G against 1.58 us over 100G, the RS-FEC that 100GBASE-R mandates outweighing the shorter serialisation. On a host whose NIC gets x8 or wider, the ceiling moves from the slot to the link. |
| Known state | Clock pinning refused by the driver. One 3080 in software thermal slowdown at 85-87 C for part of the measurements, 1719-1840 MHz against 1920 MHz on the identical card. |

This is an unfavourable configuration on every interconnect axis; the figures throughout are a
lower bound for the features, not a projection of them. Stated once, not repeated per row.

## Core concepts

- **Asymmetric / uneven TP** (`--rank-tp-ratio`): unequal per-rank attention-head and weight shard
  sizes within one tensor-parallel group.
- **Asymmetric / uneven DCP** (`--rank-kv-ratio`): capacity-weighted per-rank KV ownership during
  decode.

## Overview matrix

Verdict tokens only; each row links to its detail section, where `partial` always names the
mechanism difference. `unverified` — the check could not be completed. `n/a` — the row's
comparison point does not apply to that engine.

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
| [20](#f20) | Session KV spill | Exp, Boot-checked | partial | partial | partial | partial |
| [21](#f21) | HTCCL cross-vendor collectives | Cross-checked | no | no | partial | partial |
| [22](#f22) | fp8 dequant fallback (W8A16) | Cross-checked* | no | no | partial | unverified |
| [23](#f23) | Turing/gfx900 without sgl-kernel | Cross-checked | no | no | partial | partial |
| [24](#f24) | SWA-DCP | Cross-checked | no | no | no | no |

---

## Model coverage

### Families and quantization formats

| Family | Architecture class | Formats loaded on hardware | Status |
|---|---|---|---|
| Qwen3.6-27B | dense, GDN/attention hybrid; 64 layers, 16 full-attention, 4 kv-heads x 256; embedded NEXTN/MTP draft | FP8 native (+ separate MTP weights; KV `fp8_e5m2`); AWQ-BF16-INT4 compressed-tensors; GGUF K-quant `Q3_K_M`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, Q8-class; unsloth dynamic `UD-Q4_K_XL`, `UD-Q6_K_XL` (+ `mmproj`), `UD-Q8_K_XL` | Boot-checked |
| Qwen3.6-35B-A3B | MoE + GDN hybrid; 40 layers, 10 full-attention, 2 kv-heads x 256, 256 experts / 8 active, `nextn=1` | FP8 e4m3-dynamic (+ separate MTP weights); AWQ-4bit g32 | Boot-checked |
| Qwen3.5-122B-A10B | MoE + GDN hybrid; 48 layers, 256 experts top-8, 3 linear : 1 full | GPTQ-Int4 g128 | Boot-checked |
| Gemma-4-31B-it | dense, SWA hybrid; 10 global : 50 sliding | int4-AutoRound g128 sym; GGUF `Q4_K_M` | Boot-checked |
| Gemma-4-26B-A4B-it | MoE, SWA hybrid; 30 layers, 128 experts, global kv=2 | compressed-tensors pack-quantized W4A16 int4 g32 | Boot-checked |
| Llama-3.1-8B-Instruct | dense GQA | bf16 | Boot-checked |
| Qwen3.5-4B | dense, GDN hybrid | fp16; FP8-dynamic compressed-tensors | Cross-checked |
| Qwen2.5-1.5B, Qwen3-0.6B, Qwen3.5-2B, Qwen-0.5B | small dense and hybrid, incl. replicated-KV geometries | bf16 / fp16 | Boot-checked |

The small models serve as solo oracles and as falsifiers for the DCP, co-location and `kv == tp`
geometries.

| Format gap | Constraint |
|---|---|
| Gemma-4 GGUF beyond `Q4_K_M` | MoE, MTP and vision paths refused in the adapter |
| Qwen3.6-35B-A3B GGUF | the `qwen35` adapter maps none of its MoE expert tensors |
| Qwen3.5-122B-A10B FP8 | pinned host pool would require ~116 GiB against 108 GiB available |
| `UD-Q8_K_XL` on the generic layout | needs mixed-dtype handling for the fused GDN `in_proj_qkvz` |

### Combinations run on hardware

Each row is one configuration and carries what was measured for it. A blank cell means the quantity
was not measured for that row. Values inside the run-to-run spread are marked in place. Repeat runs
of one setup are averaged, with the count inline; separate rows are used only where a run differs
decisively — topology, format, speculative state, or a feature on against off.

#### Qwen3.6-27B FP8

| Configuration | Status | Throughput | `max_total_num_tokens` | Sessions | Other measured |
|---|---|---|---|---|---|
| TP=3 on 5090 + 2x 3080, uneven TP, uneven DCP, NEXTN/MTP k=3, flashinfer, CUDA graphs, no additional flags | Boot-checked | — | 98,328 | 1 | per-rank budgets `[26107,18280,18280]` MiB; ownership `[18,23,23]`; accept 3.69 / 2.82 / 3.28 |
| same, bench boot | Boot-checked | — | 886,336 | 1 | — |
| same, capacity KV split `[2,3,3]` | Boot-checked | 78.27 / 69.01 / 73.26 tok/s code / prose / mixed; round rate 26.298 / 26.101 / 25.900 per s | 98,328 | 1 | — |
| same, capacity KV split `[2,3,3]` | Boot-checked | 149.28 / 121.78 / 130.44 tok/s aggregate; round rate 46.686 / 46.089 / 46.382 per s | 98,328 | 2 | — |
| same, bandwidth KV split `[2,1,1]` | Boot-checked | 91.33 / 66.81 / 74.42 tok/s; round rate 27.035 / 25.831 / 26.155 per s — in the spread against `[2,3,3]` | 98,328 | 1 | over the six single- and dual-session points: round rate mean +0.67%, sd 1.74%, range -1.62 to +2.80%; tok/s mean +3.65%, sd 7.17% |
| same, bandwidth KV split `[2,1,1]` | Boot-checked | 159.09 / 123.27 / 129.19 tok/s aggregate; round rate 47.715 / 46.403 / 45.631 per s — in the spread | 98,328 | 2 | — |
| TP=3 uneven DCP, flashinfer, bs=1, ctx 131072, speculation off with `ignore_eos`, 120,420 resident tokens, capacity split `[2,3,3]` | Boot-checked | step 28.530 ms, depth term 2.2955 ms (mean of 3 cold boots; A-vs-A floor 1.07%) | 393,228 | 1 | — |
| same, bandwidth split `[2,1,1]` | Boot-checked | step 27.813 ms (-2.51%), depth term 1.7324 ms (-24.5%, t = -17.85) (mean of 3 cold boots) | 393,228 | 1 | raw KV headroom +2.3% |
| TP=3 uneven DCP, MTP, ctx 32768, capacity split | Boot-checked | 43.307 ms per verify round | 98,328 | 1 | KV rows 24,584 / 36,876 / 36,876; accept 3.871 over 155 verifies |
| same, bandwidth split, vector `[34,15,15]` | Boot-checked | 41.523 ms per verify round (-4.12%, +4.3% tok/s); depth term -26.7% | 98,328 | 1 | KV rows 52,258 / 23,055 / 23,055; accept 3.871 over 155 verifies; raw KV headroom 842,856 -> 422,480 |
| TP=3, `--rank-kv-ratio coupled`, context at model maximum | Boot-checked | — | 443,904 (`[30,17,17]`) | 1 | free VRAM at pool end 5.21 / 2.33 / 3.58 GB |
| TP=3, `--rank-kv-ratio capacity`, context at model maximum | Boot-checked | within ±1% of `coupled` at shallow, 8k and 24k depth — in the spread | 563,456 (`[33,13,18]`), +26.9% | 1 | free VRAM at pool end 2.71 / 2.46 / 2.33 GB |
| TP=3 uneven DCP, fp8 KV, CUDA graphs, bs=1, no speculation | Boot-checked | 40.3 / 40.2 tok/s code / prose | — | 1 | board power 729 W |
| same + MTP with adaptive draft length | Boot-checked | 90.7 / 116.3 tok/s (2.25x / 2.89x) | — | 1 | accept 3.32; board power 640 W |
| same + MTP, bs=16 | Boot-checked | 427 tok/s | — | 16 | — |
| TP=3 uneven DCP, MTP, default vocab split | Boot-checked | 90.30 tok/s single code (mean of 19) | 98,328 | 1 | — |
| same, `--rank-vocab-ratio 7,3,3` | Boot-checked | 95.19 tok/s single code, +5.41% (mean of 8); dual round rate +0.45 / -0.11 / +0.76% — in the spread | 98,328 | 1, 2 | — |
| same, speculative k=4 against k=3 | Boot-checked | round rate -6.6 to -7.7% single, -13.3% dual; net tok/s +0.2 to +3.5% single, -6.00% dual | 98,331 | 1, 2 | accept +7.2 to +12.2% |
| TP=3 uneven DCP, MTP, flashinfer, CUDA graphs, per-rank trace | Boot-checked | collective share of the decode span 252.2 of 1600 ms (15.8%) | — | 1 | bf16 all-reduce 27.7 us against a 31-37 us back-to-back floor |
| same | Boot-checked | collective share 415.9 of 1760.9 ms (23.6%) | — | 2 | — |
| TP=3 uneven, NEXTN k=3, session KV spill, pool 4200, prompt 1200, holder 1000 / victim 1400 new tokens, restore margin 1024, hysteresis 40 | Exp, Boot-checked | victim per verify round 41.7 ms / 76.5 tok/s on device; 113.5 / 8.8 at host floor; 37.9 / 69.8 restore transient incl. MTP backfill; 37.4 / 75.6 settled (mean of 3 boots, sd <= 1.5%) | — | 2 | victim 1400 tokens in 28.6 s; post-restore victim 76.98 / holder 76.06 tok/s; restored in 3 of 3 boots |
| same, spill never restored | Exp, Boot-checked | victim per verify round 131.5 ms / 7.6 tok/s during spill; 91.5 / 10.9 settled alone on an otherwise empty GPU (mean of 3 boots) | — | 2 | victim 1400 tokens in 71.2 s; restored in 0 of 9 boots |
| TP=3 uneven DCP, NEXTN topk 1, newest-by-arrival victim spilled, anti-starvation floor 8 | Exp, Boot-checked | spilled 2.75-2.77 tok/s; device-resident 53.8-57.9 tok/s against a ~85 tok/s solo baseline | — | 3 | device session retains 63-68% of solo; victim maximum inter-token gap 0.41 s; overlap occurred in 7 of 12 and 10 of 13 runs |
| same, static tick 1 | Exp, Boot-checked | spilled 7.50 tok/s; device-resident 23.4 tok/s | — | 3 | — |
| two-session spill choreography, headroom `H = P - 2p`, H = 1800 | Exp, Boot-checked | — | — | 2 | admission boundary between 2400 admitted and 2500 refused, i.e. `H/0.73` |
| 3 cards, context uncapped at the model maximum 262,144, demand-driven mamba sizing at 7 slots | Boot-checked | — | 883,584 | 1 | free VRAM left over 8.40-10.50 GB; pool unchanged at `max_running_requests` 1 against 8 (149,437 against 146,024) and at ctx 32,768 against 262,144 |
| same, stock flags, PP=3 even, 9 fixed mamba slots | Boot-checked | 28.28 tok/s decode; 1357.4 tok/s prefill | 146,024 | 1 | free VRAM left over 5.19 / 20.06 GB |
| same, stock flags, PP=3 uneven, 14 fixed mamba slots | Boot-checked | 35.73 tok/s decode; 1495.0 tok/s prefill | 176,066 | 1 | free VRAM left over 6.58 / 16.10 GB |
| TP=3 full feature set, 1172-token prompts, `cached_tokens=0` | Boot-checked | 91.92 tok/s decode; 1155.9 tok/s prefill at M=1, 1221.6 at M=8 (+5.7%) | — | 1, 8 | accept 3.130 |
| TP=3, shackled identically to stock flags (no MTP, no overlap scheduler) | Boot-checked | 33.46 tok/s decode; 1426.7 tok/s prefill | — | 1 | 6.8% behind the uneven pipeline split on the parallelism axis alone |
| stock flags, PP=3 even, 1172-token prompts | Boot-checked | 2597.6 tok/s aggregate prefill at M=8 | — | 8 | — |
| stock flags, PP=3 uneven, 1172-token prompts | Boot-checked | 3000.8 tok/s aggregate prefill at M=8 (+101% over M=1) | — | 8 | — |
| solo 5090, TP=1 | Boot-checked | — | — | — | no placement, out of memory |
| TP=4 across two hosts over RDMA, uneven TP 6,4,4,2, uneven DCP, NEXTN 3 + solo draft, Triton, eager, ctx 8192, bs=1, unpipelined transport | Boot-checked | 16.38 / 15.67 / 16.54 slope tok/s code / prose / mixed (mean of 3) | — | 1 | accept 3.08 / 3.03 / 2.91; per-rank utilization 10-13% main rig, 50-55% on the 2080 Ti |
| same, pipelined transport | Boot-checked | 17.31 / 17.57 / 18.36 slope tok/s (+5.7 / +12.1 / +11.0%) (mean of 3) | — | 1 | accept 3.08 / 3.03 / 2.91 |
| same, pipelined + consumer-side async overlap | Boot-checked | 17.92 / 17.12 / 17.84 slope tok/s (+3.5 / -2.6 / -2.9%) — in the spread (mean of 3) | — | 1 | accept 3.08 / 3.03 / 2.91 |
| same, pipelined + KV split toward the fastest card, 2080 Ti share 26.6% -> 12.5% | Boot-checked | 18.11 / 16.84 / 18.15 slope tok/s — in the spread (mean of 3) | — | 1 | — |
| TP=4 across two hosts, uneven TP 3,2,2,1, decode by streaming, RDMA | Boot-checked | 28.4 / 28.6 / 27.1 tok/s | — | 1 | per-rank utilization 10-13 / 19-24 / 30-35%; uneven TP inert on both wires |
| same, 1 GbE | Boot-checked | 7.8 / 7.7 / 7.5 tok/s | — | 1 | barrier 146.63 us; RDMA leads 14x at the barrier and 1.2x at 4 MiB |
| TP=3, DFLASH, split draft placement, 450-token decode | Boot-checked | 74.88 / 64.66 tok/s code / prose (mean of 4) | — | 1 | draft KV 10.9 / 22.9 / 10.1 GB across the three cards |
| TP=3, DFLASH, solo draft placement, 450-token decode | Boot-checked | 82.71 / 67.58 tok/s code / prose (mean of 4) | — | 1 | — |
| TP=3 uneven, NEXTN, greedy, 1024 decode tokens | Boot-checked | 118.8 tok/s at ctx 4096; 95.3 at ctx 49152 (mean of 2) | — | 1 | — |
| TP=4 co-located, DFLASH, greedy, 1024 decode tokens | Boot-checked | 125.7 tok/s at ctx 4096 (+6%); 98.6 at ctx 49152 (+3.5%) — in the spread at 49152 (mean of 2) | — | 1 | DFLASH runs 18.9-20.9% behind NEXTN in a multiturn regime |
| TP=3, cross-algorithm bandit, one regime cell | WIP | 75.52 tok/s | — | 1 | per-switch cost ~2.5 ms |
| TP=3, static winner of the same regime cell | Boot-checked | 89.22 tok/s | — | 1 | — |
| TP=3, cross-algorithm lazy CUDA-graph capture | Boot-checked | — | — | 1 | 542.0 MiB released by inactive speculative-depth branches |
| TP=3, DCP communication-fusion variants | Boot-checked | 80.5 / 80.6 / 80.85 tok/s — in the spread | — | 1 | — |

#### Qwen3.6-27B AWQ-BF16-INT4

| Configuration | Status | Throughput | `max_total_num_tokens` | Sessions | Other measured |
|---|---|---|---|---|---|
| TP=3, uneven TP, INT4 group alignment | Boot-checked | — | 563,763-798,528 across boots | 1 | — |
| TP=3, uneven TP/DCP + HiCache, host-RAM L2 and file L3 | Boot-checked, restore deterministic | — | — | 8 | 8 of 8 requests hit |

#### Qwen3.6-27B GGUF

| Configuration | Status | Throughput | `max_total_num_tokens` | Sessions | Other measured |
|---|---|---|---|---|---|
| `Q6_K_XL` + MTP, TP=2 on 2x 3080, legacy K-quant dispatch | Boot-checked | 67.86 / 54.17 tok/s code / prose | — | 1 | — |
| `Q6_K_XL` + MTP, TP=2 on 2x 3080, tuned K-quant kernels | Boot-checked | 88.38 / 72.22 tok/s (+30% / +33%) | — | 1 | argmax identical on 100 of 100 probes, without bit parity |
| `Q6_K_XL` + MTP, TP=3 uneven, tuned K-quant kernels | Boot-checked | 118.01 / 98.62 tok/s code / prose | — | 1 | — |
| `UD-Q6_K_XL` + `mmproj`, TP=3 uneven, NEXTN/MTP, CUDA graphs | Boot-checked | — | — | 1 | — |
| `UD-Q6_K_XL`, TP=4 co-located on 3 cards | Boot-checked | — | — | 1 | — |
| Q8-class, TP=3 uneven, 30 s window, greedy, MMQ decode threshold off | Boot-checked | 201.60 / 201.87 / 201.33 tok/s aggregate code / prose / mixed; per-request p50 25.2, p95 25.5 | — | 8 | kernel calls per rank 0 MMQ / 11320 MMVQ on every rank |
| Q8-class, same, MMQ decode threshold on | Boot-checked | 222.93 / 221.46 / 221.33 tok/s (+10.6 / +9.7 / +9.9%); per-request p50 27.9, p95 28.3 | — | 8 | kernel calls 11320 MMQ / 0 MMVQ on the sm120 rank, 0 / 11320 on the two sm86 ranks |
| `Q3_K_M`, TP=3 uneven, cold start | Boot-checked | — | — | 1 | ~50 s to ready; skippable transform stage ~44 s |
| `Q3_K_M`, TP=3 uneven, hibernate restore across process exit | Boot-checked | — | — | 1 | 8-14 s to ready; transform stage a few seconds |
| TP=3, weightless-KV lane, ownership `[6,5,5]` | Cross-checked against a TP=1 solo oracle | — | 67,000 | 1 | free VRAM 4.03 GB on the weight-holding head, 14.59 GB on the weightless workers |

#### Qwen3.6-27B, format comparison under concurrency

TP=3 uneven auto-performance, ctx 8192, CUDA graphs, one fixed 8-prompt set of a single content
class, same pool in every arm.

| Configuration | Status | Throughput | `max_total_num_tokens` | Sessions | Other measured |
|---|---|---|---|---|---|
| FP8, speculation off | Boot-checked | 37.8 / 81.7 / 156.7 / 270.7 tok/s aggregate; scaling 7.2x | 81,960 | 1 / 2 / 4 / 8 | weights 29.7 GB, 35.8 GB with the draft |
| GGUF `UD-Q8_K_XL`, speculation off | Boot-checked | 50.2 / 91.3 / 157.3 / 203.3 tok/s aggregate; scaling 4.1x | 81,960 | 1 / 2 / 4 / 8 | weights 35.1 GB |
| GGUF `UD-Q4_K_XL`, speculation off | Boot-checked | 65.2 / 106.8 / 153.0 / 189.3 tok/s aggregate; scaling 2.9x | 81,960 | 1 / 2 / 4 / 8 | weights 18.3 GB, 18.8 GB with the draft |
| FP8, speculation on | Boot-checked | 74.8 / 62.5 / 53.9 / 41.7 tok/s per session | 81,960 | 1 / 2 / 4 / 8 | accept 2.37-2.63 |
| GGUF `UD-Q8_K_XL`, speculation on | Boot-checked | 87.3 / 58.3 / 35.5 / 29.3 tok/s per session | 81,960 | 1 / 2 / 4 / 8 | accept 2.70-3.18 |
| GGUF `UD-Q4_K_XL`, speculation on | Boot-checked | 87.8 / 54.7 / 31.3 / 26.8 tok/s per session | 81,960 | 1 / 2 / 4 / 8 | accept 2.70-3.06 |

#### Qwen3.6-35B-A3B

| Configuration | Status | Throughput | `max_total_num_tokens` | Sessions | Other measured |
|---|---|---|---|---|---|
| FP8, TP=3 with 2 kv-heads: replicated KV on all 10 global layers + token-sharded DCP, MTP, CUDA graphs | Boot-checked | — | — | 1 | — |
| FP8, TP=3, `--rank-kv-ratio coupled` | Boot-checked | — | 1,911,488 pre-cap | 1 | — |
| FP8, TP=3, `--rank-kv-ratio capacity` | Boot-checked | within ±1% of `coupled` at shallow, 8k and 24k depth — in the spread | 2,187,648 pre-cap, +14.4% | 1 | needles at ~7.1k and ~17.7k retrieved in every mode |
| FP8, TP=3 + HiCache file L3 | Boot-checked; restore Cross-checked | — | — | 1 | greedy ids identical cold against restored |
| FP8, TP=3 mixed-architecture: fp8 fused MoE on sm120, Marlin W8A16 fallback on the sm86 ranks | Boot-checked | — | — | 1 | kernel-level cosine >= 0.99998 |
| AWQ-4bit, TP=3 + weighted DCP, fp8 KV, ctx 131072, chunked prefill 2048, flashinfer, no CUDA graphs | Boot-checked | time to first token 0.49 / 2.27 / 9.73 / 42.2 s at 2,048 / 8,192 / 32,768 / 122,880 prompt tokens (mean of 2-3, spread < 3%); warm prefill 4,022 / 3,717 / — / 3,117 tok/s; decode 15.9 tok/s at 2k ctx, 15.6 at 32k | — | 1 | — |
| AWQ-4bit, solo prefill at TP=1 on the 5090 + decode at uneven TP=3 `1,3,3` weighted DCP, same fp8 KV and ctx | Boot-checked; byte-identical to the same build with disaggregation off | time to first token 0.15 / 0.46 / 3.44 / 18.9 s (3.3x / 4.9x / 2.8x / 2.2x) (mean of 2-3); solo prefill 20,160 / 16,100 / 10,520 / 8,090 tok/s; decode -13% at 2k ctx, -2% at 32k | — | 1 | proxy and handoff 40-60 ms; KV re-scatter 10.1 KiB per token, 1.2 GB for a 120k prompt, ~0.2 s of PCIe |
| same, 32k prefill burst issued during a 512-token decode | Boot-checked | decode 29.27 -> 31.25 s (+6.7%); prefill burst 3.44 -> 3.54 s | — | 2 | — |
| AWQ-4bit, TP=3, uneven TP + uneven DCP + MoE expert offload | Cross-checked against a TP=1 run | — | — | 1 | 32 of 32 tokens identical |

#### Qwen3.5-122B-A10B GPTQ-Int4

| Configuration | Status | Throughput | `max_total_num_tokens` | Sessions | Other measured |
|---|---|---|---|---|---|
| solo 5090, resident fraction 0.15 | Boot-checked | 4.8 tok/s | — | 1 | — |
| TP=3 on 5090 + 2x 3080, 72 GB aggregate VRAM, with 108 GiB host RAM, resident fraction 0.25, eager | Boot-checked | 6.97 tok/s (+45%) | — | 1 | 64 resident + 16 scratch of 256 experts per layer, 176 offloaded; self-deterministic 5 of 5 |
| same, graph-static | Boot-checked | 10.61 tok/s | — | 1 | — |
| same, graph + hot-set | Boot-checked | 16.34 tok/s | — | 1 | — |

#### Gemma-4

| Configuration | Status | Throughput | `max_total_num_tokens` | Sessions | Other measured |
|---|---|---|---|---|---|
| 31B-it int4-AutoRound, TP=1 and TP=3 uneven, Triton, bf16 KV | Boot-checked | — | — | 1 | — |
| 31B-it int4-AutoRound, TP=3 uneven + EAGLE3 | Boot-checked | — | — | 1 | 4 of 4 temp-0 probes byte-identical to a no-speculation run; not a valid cross-check, see legend |
| 31B-it int4-AutoRound, TP=3, SWA-DCP with `--swa-pool-sizing cap`, CUDA graphs | Cross-checked against a TP=1 solo-5090 oracle | — | — | 1 | needle planted ~3k tokens beyond the 1024-token window retrieved byte-identically |
| 31B-it GGUF `Q4_K_M`, TP=1 on the 5090 | Boot-checked | ~61 tok/s | — | 1 | self-deterministic |
| 31B-it GGUF `Q4_K_M`, TP=3 uneven | Boot-checked | — | — | 1 | — |
| 26B-A4B-it W4A16, TP=1 | Boot-checked | — | — | 1 | 3.6k needle retrieved |

#### Llama-3.1-8B bf16

| Configuration | Status | Throughput | `max_total_num_tokens` | Sessions | Other measured |
|---|---|---|---|---|---|
| TP=2, 5090 head + 3080 weightless worker, weightless-KV lane, CUDA graphs, ctx 2048, no speculation | Cross-checked against a TP=1 solo oracle | 71.67 / 69.52 / 71.80 / 71.53 tok/s one_token / code / prose / mixed; 13.95 / 14.38 / 13.93 / 13.98 ms per decode step (mean of 2 cold boots) | 16,384, configured cap | 1 | — |
| same + EAGLE3 chain speculation, topk 1, 3 steps, 4 draft tokens, solo draft | Boot-checked | 80.67 / 120.05 / 87.20 / 86.18 tok/s (+12.6 / +72.7 / +21.5 / +20.5%); 17.06 / 17.62 / 17.69 / 17.27 ms per verify round (mean of 2 cold boots) | 16,384, configured cap | 1 | accept 1.376 / 2.116 / 1.542 / 1.488; a verify round costs 1.22-1.27 decode steps and returns 1.38-2.12 tokens |
| same lane, decode eager | Boot-checked | 13.1 tok/s | — | 1 | — |
| same lane, decode graph-captured | Boot-checked | 63.5 tok/s | — | 1 | — |
| same lane, intra-rig DCP collective overlap | Boot-checked | -0.17 to +3.12% across the four content classes — in the spread | — | 1 | — |
| TP=5 across two hosts, ratio 4,3,3,2,1 over 5090 / 3080 / 3080 / 2080 Ti / Vega 64, ctx 4096, eager, host-staged transport, no DCP, no speculation | Boot-checked; prose byte-identical to solo | 4.32 / 4.73 / 4.82 tok/s decode; 61-63 tok/s prefill | capped at 4096 | 1 | weights per rank 4.41 / 3.55 / 3.55 / 2.46 / 1.66 GB |
| same, EAGLE3 split draft, ctx 2048 | Boot-checked | 4.88 / 4.58 / 5.39 tok/s decode; 59-60 tok/s prefill — in the spread against solo draft | 132,871 | 1 | accept 1.36 |
| same, EAGLE3 solo draft, ctx 2048 | Boot-checked | 4.39 / 5.21 / 4.99 tok/s decode; 62-63 tok/s prefill — in the spread | 133,802 | 1 | accept 1.36 |
| solo 5090, TP=1 | Boot-checked | 76 tok/s decode | — | 1 | — |

#### Qwen3.5-4B and small vehicles

| Configuration | Status | Throughput | `max_total_num_tokens` | Sessions | Other measured |
|---|---|---|---|---|---|
| Qwen3.5-4B fp16, mixed-vendor TP=2 on 2080 Ti + Vega 64, Triton, eager, even 2/2, host-staged `gloo` | Cross-checked against `torch.distributed` | 10.28 tok/s decode; 540.5 tok/s prefill | — | 1 | — |
| same, even 2/2, on-GPU `device` transport | Cross-checked; byte-identical to `gloo` | 14.07 tok/s decode (+37%); 786.2 tok/s prefill (+45%) | — | 1 | — |
| same, uneven 3,1, `gloo` | Cross-checked | 11.13 tok/s decode; 604.7 tok/s prefill | — | 1 | — |
| same, uneven 3,1, `device` | Cross-checked; byte-identical to `gloo` | 16.51 tok/s decode (+48%); 982.0 tok/s prefill (+62%) | — | 1 | — |
| Qwen3.5-4B-FP8-dynamic, mixed-vendor TP=2, W8A16 dequant fallback | Cross-checked; byte-identical to the solo oracles | 12.67 tok/s decode (-23%); 966 tok/s prefill | — | 1 | — |
| Qwen3.5-4B-FP8-dynamic, solo 2080 Ti | Cross-checked | 15.23 tok/s decode | — | 1 | — |
| Qwen3.5-4B-FP8-dynamic, solo Vega 64 | Cross-checked | — | — | 1 | weights 6.27 GB with 1.07 GB free; fp16 needs 8.8 GB and has no placement |
| Qwen3.5-4B fp16, solo Vega 64, external Triton port | Boot-checked | 10,189-token prompt prefilled in 5.4 s, 1879 tok/s | — | 1 | the previous torch-native path ran out of memory from ~4k |
| Qwen-0.5B fp16, solo 2080 Ti, greedy | Boot-checked | 48.9 tok/s | — | 1 | — |
| Qwen-0.5B fp16, solo Vega 64, greedy | Boot-checked | 25.7 tok/s | — | 1 | — |
| Qwen2.5-1.5B, TP=4 / DCP=2 | Boot-checked | — | 525,897 | 1 | — |
| Qwen3-0.6B, Qwen3.5-2B, DCP and co-location falsifiers across Triton/flashinfer and NCCL/gloo | Boot-checked | — | — | 1 | — |

**Without a hardware boot:** fast-lane priority scheduling (`Built` only); solo draft placement,
which has no dedicated boot and runs only inside the weightless-lane, cross-host TP=4 and TP=5
configurations above; cross-vendor CUDA-graph capture; MoE-model hibernation (not built); a session
KV spill landing in the same round as a drafter-in-tick step; more than one simultaneously spilled
session (the mechanics are unit-tested; boots show one spilled session among up to three
co-resident ones); tree speculation at
`--speculative-eagle-topk > 1` under weighted DCP (gated off); the `fp8.py` `Fp8Config` family on
the capability probe. Gemma-4 does not run under uneven DCP, since `SWAKVPool` carries no DCP mask,
and it refuses the flashinfer backend. Throughput under SWA-DCP has not been taken.

### Measurement

Raw tok/s tracks output content on this rig (r = 0.90) and carries a 2.6-4.2% boot-to-boot spread —
2.60% is the worst directly measured excursion — against 0.09-0.85% for milliseconds per verify
round, so any claim finer than roughly 3.5% between two arms is stated on the round-time axis, and
the detection limit is established A-vs-A before the arms are compared. Decode is taken as a slope
over two generation lengths at one prompt, or by streaming wherever a repeated prompt would be
served from the prefix cache. Values inside the spread are marked and are not counted as gains. Two
limits apply to individual rows: the concurrency sweep covers one content class, so only its
arm-against-arm statement is load-bearing; and the energy figures are single runs measuring GPU
board power alone. Transport-level and collective-level microbenchmarks, which involve no model,
are in section 21.

---

## Detail sections

<a id="f1"></a>
### 1. Asymmetric tensor parallelism

`--rank-tp-ratio auto` — unequal per-rank shard sizes within one TP group. **Boot-checked.**

| Item | Detail |
|---|---|
| Validated | TP=3 on 5090 + 2x 3080, Qwen3.6-27B FP8; greedy decode byte-identical run-to-run and cold-vs-warm on the same GPUs — not a cross-hardware claim, see row 11 |
| Correctness fix | `o_proj`-vs-head-split reject guard for 3 attention classes whose attention silently used the wrong head split |
| Correctness fix | DFLASH per-rank attention and MLP shards, validated green at MLP units `[68,34,34]` |
| Refuted | replicated-KV widening to `kv == tp`: implemented, GPU-measured, reverted — see Guarded / descoped |

**Upstream:** replaces sglang's requirement of equal, head-divisible shards.

<a id="f2"></a>
### 2. Asymmetric decode context parallelism

`--rank-kv-ratio` — capacity-weighted per-rank KV ownership during decode. **Cross-checked.**

| Check | Setup | Result |
|---|---|---|
| Triton uneven DCP against a DCP-off ground truth | 27B FP8, TP=3, greedy, no speculation | `short_code` byte-identical arm for arm. The same run's 11,650-token `chunked` prompt also matched, but sits past the fp8@3080 boundary and is excluded from the tier |
| Triton against flashinfer, chain speculative verify under uneven DCP | 27B FP8, TP=3, MTP, greedy, CUDA graphs, 4 prompts | token ids identical arm for arm on the 3 short prompts, `meta_info.spec_accept_length` in the same band. The 11,650-token prompt is separately on record as cache-state-sensitive on the Triton lane and is excluded |

The arg gate requiring a non-uniform `--rank-tp-ratio` is genuine, not arbitrary. An explicit token
vector with no plan formerly booted green while doing nothing, and now hard-rejects
(red-then-green). Stock `--dcp-size N` under the Triton backend silently corrupts output when KV
heads are not replicated across the DCP group; the fork's own uneven-DCP geometry is exempt.

**Upstream:** replaces sglang's DCP, which only splits KV evenly across ranks.

<a id="f3"></a>
### 3. Rank-to-GPU mapping and co-location

`--rank-gpu-id`, `--rank-gpu-memory-mib` — assigns each rank to an NVML-resolved physical GPU;
duplicates co-locate multiple ranks on one GPU. **Boot-checked.**

| Item | Detail |
|---|---|
| Exercised | TP=4 co-located on 3 cards, Qwen3.6-27B `UD-Q6_K_XL`; also underpins row 18 |
| Requirement | NCCL >= 2.30, shipped in the fork's container image |
| Decoupling | `--rank-tp-ratio` / `--rank-kv-ratio` no longer require `--rank-gpu-id`: sharding-ratio validity and physical placement are independent concerns, and coupling them blocked the cross-vendor case, where NVML cannot name an AMD rank |

**Upstream:** sglang places ranks via `CUDA_VISIBLE_DEVICES` only.

<a id="f4"></a>
### 4. Solo drafter placement

`--speculative-draft-placement solo` — runs the draft model unsharded on one GPU, broadcasting its
output instead of all-reducing. **Built** — registered unit tests for solo placement, weight/KV
planning and vocab broadcast; no dedicated hardware boot. It runs inside the weightless-lane,
cross-host TP=4 and TP=5 configurations.

**Upstream:** no equivalent flag in sglang.

<a id="f5"></a>
### 5. Cross-algorithm drafter routing

`--speculative-cross-algorithm*` — NEXTN/MTP and DFLASH resident simultaneously, switched per batch
by a bandit controller on accept-tokens/round, rank-0 decision plus TP broadcast. **WIP.**

| Item | Detail |
|---|---|
| Implemented | dual residence, per-batch switching, bandit controller with a registered test |
| Missing | the context-length gate from the drafter training config |
| Validated | lazy single-graph capture and DFLASH context retirement, green under CUDA graphs; 542.0 MiB released |
| Measured | the bandit loses its regime cell against the static winner, 75.52 against 89.22 tok/s; per-switch cost ~2.5 ms |

**Upstream:** no equivalent in sglang, which adapts or selects a single drafter's parameters.

<a id="f6"></a>
### 6. CUDA graph memory aliasing for spec branches

Inactive speculative-depth CUDA-graph branches hold no physical VRAM, via cuMem tag aliasing
(`kv_vmm_backing` / adaptive runtime state). **Boot-checked** — the recorded GPU figure is
542.0 MiB released under CUDA graphs on the lazy-capture arm.

**Upstream:** sglang has related VMM/cuMem machinery, not applied to speculative CUDA-graph
branches.

<a id="f7"></a>
### 7. MoE expert offload + asymmetric TP/DCP

MoE expert offloading to host RAM combined with asymmetric TP and DCP (GPTQ/AWQ/FP8).
**Boot-checked** on a 122B-A10B across three mismatched GPUs; **Cross-checked** on 35B-A3B AWQ, at
32/32 tokens identical to a TP=1 run. Offloaded output is self-deterministic but not bit-identical
to the no-offload case, since Marlin-Int4 tiling reduces in a different order. Numbers above.

**KV-pool reclaim.** The weight VRAM the offload frees is claimed by the KV pool. No second
sizing path exists and none is needed: the KV budget is profiled from a live free-memory reading
taken after the weights are resident, so the reclaim lands in it by construction — provided the
release has happened, on every rank, before anyone measures. #123's eager install (right after
`load_model`, ahead of pool sizing) supplies the first half; #119 adds the parts that make it hold:

- an **ordering invariant** — a FusedMoE layer still waiting to install when the pool is sized is a
  hard error, not a silent fallback to the pre-#123 behaviour where the pool was sized against the
  pre-offload footprint (the #77 "known limitation");
- a **group-ordered release** — `gc.collect() -> empty_cache() -> barrier` before any rank reads
  `mem_get_info()`. That reading is driver-level and therefore sees co-located siblings, while the
  caching allocator only returns freed blocks at `empty_cache()`; unsynchronized, a rank that reads
  early counts a sibling's already-released expert weights as still occupied. Below the offload the
  skew was small — at f=0.25 on the 122B run it is the whole reclaim (~18 GiB on the shared card);
- a **rank-uniform verdict** — "the reclaim is present" is a MIN over the TP group, so a rank that
  released nothing (MoE-free shard, failed install) makes the whole group take the plain path
  rather than half the ranks synchronizing while the other half measures unsynchronized;
- **accounting** — released device/host bytes are tallied per rank and logged next to the resulting
  KV budget (`[offload-kv-regain]`), so the win is readable from a boot log.

No new budget term is introduced, so the #68 graph-capture reserve is untouched: the reclaim
arrives as free bytes and is spent net of the same reserves as any other free byte. Switch:
`SGLANG_MOE_OFFLOAD_KV_REGAIN` (default on, additionally gated on
`SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0`, so the no-offload path is byte-identical).

**Upstream:** SGLang/vLLM offload weights layer-granularly (`--cpu-offload-gb`), not
expert-granularly, and not combined with asymmetric TP/DCP (partial). llama.cpp/ik_llama.cpp have
the same expert-granular idea (`-ot`/`-ncmoe`/`--n-cpu-moe`; ik_llama.cpp also runs its own
`iqk_mul_mat` kernel lineage, see row 8d) but nothing to combine it with, since neither
asymmetric-TP nor DCP exists there (partial).

<a id="f8a"></a>
### 8a. Bespoke GGUF adapter framework

`gguf_registry` + `GGUFAdapterBase` — per-model-family GGUF loaders (name maps and inverse weight
transforms) on top of the generic GGUF path, plus sibling-file config/tokenizer loading for
architectures the generic metadata reader cannot parse. **Boot-checked** — registry with two
families, unit tests for header and sizing; boot evidence comes from rows 8b-8f, which load
through it.

**Upstream:** sglang's generic GGUF path cannot load these architectures.

<a id="f8b"></a>
### 8b. Qwen3.5/3.6 GGUF

GGUF arch `qwen35` / `qwen35moe` — GDN/RMSNorm/`out_proj` inverse transforms, plus NEXTN/MTP draft
including MoE draft, loaded from the same file. **Boot-checked** — dense, MoE and NEXTN/MTP;
K-quants `Q4_K_M`…`Q8_0` coherent and greedy-deterministic; `Q6_K` validated at asymmetric TP=3.

**Upstream:** unsupported in sglang.

<a id="f8c"></a>
### 8c. Gemma-4 GGUF

GGUF arch `gemma4`, dense — inverse transforms distinct from Qwen: dequantized `token_embd`,
identity norm handling, tied `lm_head`, `k==v` shard duplication. **Boot-checked** — Gemma-4-31B-it
`Q4_K_M`, TP=1 on the 5090 at ~61 tok/s, coherent and self-deterministic; asymmetric TP=3 green.
MoE, MTP and vision fail fast; only `Q4_K_M` is verified.

**Upstream:** unsupported in sglang.

<a id="f8d"></a>
### 8d. GGUF K-quant compute kernels

`sgl-kernel` MMQ/MMVQ — per-device MMVQ↔MMQ crossover, prefill-oriented MMQ cap, batched MMVQ,
quantized vocab/embedding, I-Matrix quant. **Boot-checked** — merged with kernel tests. The
crossover is opt-in via `--gguf-mmq-decode-threshold`, default off; it is not byte-identical when
on, since MMQ and MMVQ reduce in a different order, and the flag off reproduces the prior dispatch
exactly. Numbers above.

**Upstream:** sglang has the base MMQ/MMVQ kernels; the crossover, cap and quantized-vocab tuning
are fork-only.

<a id="f8e"></a>
### 8e. Asymmetric-TP x GGUF correctness

Composes GGUF with row 1 — K-quant superblock alignment, GDN/MoE per-rank block coarsening,
GGUF-MoE out-of-bounds expert-id fixes, per-rank local-expert-count guard, and the same alignment
applied to compressed-tensors AWQ/GPTQ INT4. **Boot-checked** — merged bugfixes with registered
tests. The out-of-bounds-expert-id and superblock-alignment class was found through real GPU
crashes and reads; each guard test corresponds to a reproduced hardware fault (red-then-green).

**Upstream:** n/a — asymmetric TP is absent from sglang, so this bugfix class does not apply there.

<a id="f8f"></a>
### 8f. Multimodal and dynamic-quant GGUF

Loads a vision tower from a companion `mmproj` GGUF, and unsloth "UD" dynamic-quant GGUFs of mixed
precision. **Boot-checked** — `UD-Q6_K_XL` with `mmproj` validated in the benchmark matrix.
`UD-Q8_K_XL` requires mixed-dtype handling for the fused GDN `in_proj_qkvz`; without it the loader
rejects the file.

**Upstream:** sglang's generic path does not load these variants for the affected architectures.

<a id="f9"></a>
### 9. Hibernate checkpoint/restore

Persists warm server state to disk so it survives process exit and reloads without full
re-initialization. **Boot-checked** for dense GGUF under asymmetric TP=3; numbers above, as a
documented range rather than a single raw A/B run. The FP8 path is functional but has no expensive
transform to skip; MoE-model hibernation is deferred.

**Upstream:** sglang has diffusion-server offload/wake-up only, no full LLM-server snapshot.

<a id="f10"></a>
### 10. Measured VRAM budget

`--rank-gpu-memory-mib` plus a component registry — per-rank absolute MiB budget derived from
measured component usage rather than a global fraction, plus a self-calibrating KV split whose boot
log emits a vector hint fed back on restart. **Boot-checked.**

**Upstream:** sglang uses a fraction-based global setting (`mem-fraction-static`), with no per-rank
absolute budget.

<a id="f11"></a>
### 11. Cross-architecture speculative determinism

Verify-sync and CUDA-graph padding across sm86 + sm120, with sampling broadcast from rank 0.
**Boot-checked** — three divergence root causes resolved; the emitted greedy token sequence is
reproducible across the mixed-architecture TP group. Activations are not bit-identical, since sm86
and sm120 reduce in a different order; agreement is enforced by the rank-0 sampling broadcast, not
by an independent per-architecture comparison.

**Upstream:** sglang has single-architecture determinism modes; mixed-GPU-architecture TP groups are
not addressed.

<a id="f12"></a>
### 12. Weightless-KV lane

`--weightless-kv-fastlane` — a meta-device worker holds only KV cache and attention while a separate
head holds the weights. Unrelated to row 16's fast lane despite the shared name. **Cross-checked** —
the determinism harness checks output against a TP=1 solo run as oracle.

| Item | Detail |
|---|---|
| In place | chunked prefill and graph-decode paths |
| Per-role KV precision | `--weightless-kv-worker-cache-dtype`, opt-in, default off: workers may hold their KV token-shard in fp8 while the head keeps its own format, since KV bytes cross the role boundary only in the model compute dtype. Whether this buys capacity depends on which rank binds the min-reduced token budget; the boot log names it |
| Chain speculation | EAGLE/EAGLE3/NEXTN at `--speculative-eagle-topk 1` composes with the lane via `--speculative-draft-placement solo` hosted on the lane's head rank |
| Refused alongside | tree verify, adaptive draft length, the block-decode/host-spill tier |
| Open | no correctness oracle for lane plus speculation; token identity against a non-speculative run is not a valid gate, see legend |

Design notes: `docs_new/weightless_kv_role_precision.md`, `docs_new/weightless_chain_spec.md`.

**Upstream:** no equivalent in sglang.

<a id="f13"></a>
### 13. Rig dashboard / planner UI

Capacity-planning tool reporting work-normalized J/token under asymmetric DCP
(`tools/rig_dashboard`). **Exp** — functional but under active development, not production-ready.

**Upstream:** n/a — external tooling.

<a id="f14"></a>
### 14. Single-node PD disaggregation

Single-node heterogeneous prefill/decode split: prefill solo on the fastest card at TP=1, decode
distributed under asymmetric TP/DCP, with GDN/Mamba state handoff. **Boot-checked** — pair green,
token-vector KV re-scatter, crash-robust, byte-identical to the same build with disaggregation off.
Numbers above.

**Upstream:** sglang provides base PD-disaggregation; the single-node solo-prefill plus
asymmetric-TP/DCP decode plus GDN handoff is the fork's delta on top of it.

<a id="f15"></a>
### 15. Asymmetric-TP quantization correctness

Asymmetric-TP quant correctness plus upstream quant bugfixes — GPTQ-MoE `w2_scales` at TP>1, AWQ
marlin zero-point staging, `moe_wna16` K-mask, compressed-tensors/AutoRound-int4 group alignment,
and a mixed-architecture fp8 MoE path with a Marlin W8A16 fallback on sm86 ranks at kernel-level
cosine >= 0.99998. **Boot-checked.** The GPTQ `w2_scales` defect, symmetric and asymmetric, was
found during the 122B MoE boot campaign and reproduced on hardware before the fix (red-then-green).

**Upstream:** sglang has the underlying quant methods but a genuine stock GPTQ-MoE TP>1 load defect,
fixed here, and no asymmetric-TP alignment.

<a id="f16"></a>
### 16. Fast-lane priority scheduling

`--enable-fast-lane` — opt-in latency-priority class that preempts a tagged request into the running
batch, with a reserved-heavy-slots floor and heavy-aging; default off. Flags:
`--fast-lane-priority`, `--fast-lane-reserved-heavy-slots`, `--fast-lane-heavy-aging-ms`.
**Built** — no hardware boot.

**Upstream:** sglang already has priority scheduling and preemption; the reserved-floor fast-lane
class is the addition.

<a id="f17"></a>
### 17. HiCache under asymmetric-TP/DCP

Makes sglang's tiered KV cache (host-RAM L2, file L3) correct under non-uniform per-rank layouts —
global-to-owned-compact index translation, an NCCL-deadlock fix, a hybrid-SWA host-pool fix.
**Boot-checked** — 8/8 concurrent requests hit, restore deterministic. The prefetch deadlock was
reproduced live before the fix (red-then-green).

**Upstream:** HiCache itself is upstream sglang; correctness under non-uniform layouts is the delta.

<a id="f18"></a>
### 18. TP greater than num_kv_heads

Replicated KV plus token sharding, letting the TP degree exceed the model's KV-head count and, via
co-location, the physical GPU count, including GQA re-grouping to single-head geometries.
**Boot-checked** — Qwen3.6-35B-A3B FP8 at TP=3 with 2 kv-heads (replicated KV on all 10 global
layers, MTP, CUDA graphs), and TP=4 co-located on 3 cards.

**Upstream:** sglang already replicates KV under GQA when `tp > kv_heads`, but not combined with
asymmetric-TP/token-sharded DCP.

<a id="f19"></a>
### 19. Broad model bring-up under asymmetric-TP

**Boot-checked**, per model.

| Model | Bring-up work |
|---|---|
| Qwen3.6-27B (GDN), Qwen3.6-35B-A3B (MoE) | asymmetric TP=3 |
| Gemma-4-31B dense | EAGLE3 head fix; `--swa-pool-sizing` |
| Gemma-4-26B-A4B MoE SWA-hybrid | boot fix — vision-ignore, gated-GeLU Marlin |
| small / replicated-KV models | oracle and falsifier vehicles |

**Upstream:** n/a — model-support work specific to the fork's own asymmetric-TP and speculative
code.

<a id="f20"></a>
### 20. Session KV spill

`--enable-kv-session-offload` plus tick/margin/hysteresis/pool flags — on VRAM overflow, the newest
in-flight session's KV shard moves to a pinned host pool while the session keeps decoding through a
bs=1 host-streamed spill tick interleaved with the device batches; FCFS-by-arrival victim order,
stock retraction as fallback. **Exp, Boot-checked** — decode-side spill, restore and prefill-side
admission all run on hardware with speculation active; spill rows in the combinations table above,
admission checks below.

| Area | State |
|---|---|
| Spill | whole-session, or a partial tail segment with hybrid device+host attention across the boundary; multiple concurrent spilled sessions (per-session host regions, `--kv-session-offload-max-spills`); host pool sized from a node-wide RAM budget (`--kv-session-offload-host-ram-gib`) |
| Decode while spilled | spill tick at a static interval or under a self-calibrating cadence with an anti-starvation floor; eager by default, a bucketed CUDA-graph tick behind `SGLANG_KVSO_SPILL_GRAPH=1` (per-rung graph==eager at machine zero) |
| Restore | FIFO with margin and hysteresis; incremental wave-back behind a configurable free-token threshold; readiness counts radix-evictable tokens, not the free list alone — a finished neighbour session returns its KV as evictable, so a free-list-only gate deadlocks precisely when the most memory is available. The commit path stays authoritative and evicts before re-checking |
| Speculation | a spilled session decodes under MTP/NEXTN: the draft-KV share spills and restores with the session, with on-device resume and draft backfill; the drafter can run inside the spill tick (`--kv-session-offload-spec-in-tick`); every spill+spec boot rides the `KVSO_ALLOW_SPEC=1` opt-in gate |
| Prefill-side spill | `--kv-session-offload-prefill`: a prompt beyond the device budget is admitted born-spilled — no device KV slots allocated — and handed to the spill tick; runs under speculative decoding and under the overlap scheduler |
| Scheduler integration | overlap scheduler supported (the spec deferred-commit hazard takes a post-verify snapshot); DFLASH rungs stay out of the spill tick |
| Bounds that hold | GDN/Mamba state stays device-resident — the KV shard and the draft share spill, the recurrent state does not; more than one simultaneously spilled session is unit-tested (victim ordering, fast-lane multi-eviction) but not yet shown on hardware — boots show one spilled session among up to three co-resident ones, and three co-resident sessions are already the ceiling of this scheduler's admission arithmetic (a flag-OFF control shows the identical ceiling); a spill landing in the same round as a drafter-in-tick step has not been observed in validation; the concurrent two-stream dispatch (`SGLANG_KVSO_DECOUPLE=1`: device batch and spill tick in the same iteration, own stream, flashinfer workspace and DCP communicator for the spill lane) runs stably but holds the serial rate — only the token-sharding collectives have their own communicator, and the two lanes serialize behind the shared tensor-parallel one, so the default tick stays time-multiplexed with the device batch |

| Configuration | Metric | Value |
|---|---|---|
| Qwen3.6-27B-FP8, uneven TP=3/DCP=3, MTP; feature armed vs off | ms per verify round | 37.74 vs 37.92, inside the 0.09-0.85 % boot-to-boot band |
| born-spilled deep prompt, 1829 tokens against a 1306-token device budget, MTP | admissions with no device KV allocation | 2 of 2, on all three ranks, handed over to the spill tick |
| same, drafter-in-tick requested without its gate | declined admissions | declined on all three ranks, naming the condition; the request is still served through the ordinary path |
| the spilled session's output | coherence | 200 tokens, valid, no repetition flags |
| spill under NEXTN k=3, restore margin 1024, hysteresis 40 | restores | 3 of 3 boots (decode figures in the combinations table) |
| host pool from the RAM budget | per-rank pool | 1.00 GB of a 24 GiB node budget |

**Upstream:** sglang retracts — frees and re-prefills — on exhaustion, rather than keeping a spilled
request decoding.

<a id="f21"></a>
### 21. HTCCL cross-vendor collectives

`SGLANG_HTCCL_TRANSPORT` = `gloo` / `shm` / `device` / `ucx` — vendor-neutral TP collectives that
never call NCCL/RCCL, so one TP group can mix an NVIDIA and an AMD GPU. `device` reduces on-GPU over
host-mapped memory and is CUDA-graph capturable; `ucx` is the cross-host data plane, with the same
host-staged semantics as `gloo` but RDMA instead of TCP. **Cross-checked**, merged.

| Check | Setup | Result |
|---|---|---|
| Known-answer tests per collective / dtype / world-size / transport | against `torch.distributed` | green. 2 real bugs found pre-cross-vendor: an output-buffer aliasing defect, and a `reduce_scatter` wrong-axis defect for `dim >= 2` (red-then-green on all three transports) |
| Cross-vendor collectives, eager | 2080 Ti sm75 + Vega 64 gfx900 | byte-exact against `torch.distributed`; model-scale byte-identical to `gloo` (Qwen3.5-4B, even 2/2 and uneven 3,1) |
| Cross-vendor with CUDA graphs | same pair | **not demonstrated.** 4 CUDA-only assumptions were found and fixed on gfx900; the last — a device-side `assert()` that fails kernel launch entirely on gfx900 — is **not merged**. Symmetric decode capture and an NVIDIA-side prefill-capture assertion remain open |

**`ucx` transport.** One registry entry plus two modules; no dispatch site changed. Serves
`all_reduce`, `all_gather`, `broadcast`, `reduce_scatter` and an internal `barrier`. Host-staged
(GPU → pinned host → UCX → pinned host → GPU), because there is no GPUDirect on this hardware.

| Design point | Detail |
|---|---|
| ctypes over `libucp`, not ucx-py or Cython | version parity requires loading a *specific* library path; `ucx-py` bundles its own UCX and hard-codes one side of the mismatch, and is asyncio-shaped, the wrong control flow for a synchronous collective on a bs=1 decode path. Cython needs a compiler and UCX headers per host, and the second rig has the runtime libraries without the `-dev` package. A subprocess bridge adds an IPC hop to a ~1.5 us path. ctypes needs no build step and no headers, and takes the library from `SGLANG_HTCCL_UCX_LIB`. Struct layouts are mask-driven, over-allocated, and the transcribed offsets are asserted at import |
| Version parity enforced, not hoped for | mixed UCX releases do not degrade; endpoint creation fails with `invalid bandwidth 0.00`. The rendezvous gathers each rank's version over the existing `gloo` `cpu_group` and refuses **before any endpoint exists**, naming every rank's version, library path and the `SGLANG_HTCCL_UCX_LIB` remedy |
| Latency shaping | default is a single-step full exchange, one round trip at any world size; `all_reduce` switches to a ring above `SGLANG_HTCCL_UCX_RING_MIB`. Endpoints are persistent and wired at construction, so no decode step pays a handshake. `handles()` is size-independent, unlike `shm`'s slot ceiling, so two ranks can never disagree about whether a collective goes over UCX or `gloo` |

**Status (`ucx`):** implemented and validated on real RDMA, CPU-only — not yet exercised with a GPU
or a model.

| Scope | Result |
|---|---|
| Local, single host, `self`/`sm`/`tcp` loopback | 16 tests green across world 2/3/4, every collective against a computed reference, including the buffer-aliasing and `reduce_scatter dim>=2` traps, forced multi-chunk transfers and idempotent teardown; the 47 pre-existing tests still pass |
| Cross-rig over 40G RoCE, `UCX_TLS=rc` with no TCP fallback, so a pass proves RDMA carried it | all collectives green on both ranks; rendezvous and wireup 0.11 s |
| Raw link the same day, `ucx_perftest tag_bw` unidirectional | 3413 MB/s (~27.3 Gbit/s) |

Transport throughput, median per direction while the reverse direction runs simultaneously, world 2,
5 repetitions of per-cell medians:

| Cell | Unpipelined | Pipelined | Delta |
|---|---|---|---|
| 8 KiB | 37.0 us / 1.77 Gbit/s | 26.6 us / 2.47 Gbit/s | -28% latency |
| 64 KiB | 60.2 us / 8.71 | 50.4 us / 10.39 | -16% |
| 512 KiB | 232.9 us / 18.01 | 223.3 us / 18.78 | -4% |
| 4 MiB | 1625.7 us / 20.64 | 1583.3 us / 21.19 | -3%, already at the wire |
| 32 MiB | 24.30 ms / 11.05 | 15.42 ms / 17.41 | -37% / +58% |
| barrier | 12 us, min 9.9 | 5.5 us | -54% |

The 4 MiB peak is ~76% of the unidirectional budget while moving traffic both ways; the 32 MiB point
moved from 53% of the 4 MiB peak to 82%, putting the transport within ~20% of the wire at large
sizes. An earlier measurement of the same cells on the same link, before the comparison above:
8 KiB 35 us / 1.8 Gbit/s, 64 KiB 75 us / 7.0, 512 KiB 355 us / 11.8, 4 MiB 1.61 ms / 20.8 (peak),
32 MiB 23.6 ms / 11.4; barrier median 12 us, min 9.9 us.

| Cost | Decomposition, profiled on the real link |
|---|---|
| 12.37 us barrier, 2000 samples; a same-host `self`/`sm`/`tcp` run gave 10.9 us, so the wire contributed essentially nothing | setup 1.43 — a `fill_(0)` on a byte nobody reads, plus a peer dict rebuilt with f-string keys per call; posting 4.29 — two ctypes crossings wrapped in eagerly formatted error strings, per-post `data_ptr`/`numel`/`element_size`, a `c_void_p` per post; waiting 4.13 — median 5-6 spin passes, each rebuilding a list and reading the clock |
| 5.5 us barrier after the change; of the original 12 us, ~7 us was bookkeeping around two library calls | setup 0.20 / posting 2.25 / waiting 3.08 — the ctypes marshalling floor for two 5-argument calls plus 5 poll passes over a ~1.5 us link |
| 32 MiB regression | four host passes over 32 MiB cost 9.4 ms single-threaded; a 32 MiB buffer runs at ~13 GiB/s against ~34 GiB/s for a 4 MiB one, since it no longer fits in L3. Against a ~13 ms wire budget this fully accounts for the 23.6 ms observed |
| 8 KiB `all_reduce`, 48 samples | stage 4.5 / post 8.1 / wait 8.3 / finish 4.2 us plus ~2 us residual (lock, seq, `empty_like`). Post and wait are the ctypes + UCX-internal + RTT floor; stage and finish are torch dispatch |

Changes: a two-parity chunk pipeline for `all_reduce`, scheduled
`stage-in(k+1) → wait(k) → post(k+1) → finish(k)`; progress interleaving every
`SGLANG_HTCCL_UCX_PROGRESS_KIB` (default 256) inside the staged copies, without which the next
chunk's rendezvous handshake cannot start during a several-hundred-microsecond memcpy — removing it
drops the 32 MiB figure from 17.4 to 13.8 Gbit/s; and a short small-message path with precomputed
barrier slots, memoised staging records, error strings built only on failure, bound ctypes function
objects, a single-request fast path in `wait`, and a dedicated single-chunk branch.
`SGLANG_HTCCL_UCX_PIPELINE=0` restores the previous path as an A/B control.

`all_gather` received the same pipeline and a single-chunk fast path; the own rank's slice is copied
device-locally inside `finish(k)` and never crosses the wire. Correctness: all references exact at
atol 0.0, ramp payloads, chunk-boundary, ragged and 2d-axis cases, pipelined == unpipelined and
fastpath == generic bit-for-bit, in the registered unit test, the local selftest at world 2 and 3,
and cross-rig over RDMA; a deliberate parity-bug mutation flips 8+ tests red.

| `all_gather` cell | Before | After pipeline | After small-message pass |
|---|---|---|---|
| 8 KiB | 43.2 us | 41.2 us | 27.0 us |
| 64 KiB | 67 us | — | 49.0 us |
| 512 KiB | 268 us | 242 us | 219 us |
| 32 MiB | 26.4 ms | 24.9 ms | — |

`all_reduce` and barrier held steady throughout: 8 KiB 27.4 → 26.5-27.1 us, barrier 5.2 us. 32 MiB
`all_gather` barely moved, and phase timers say why — total 22.5 ms = stage 1.4 + wait 8.1 + finish
12.6. The finish pass writes 64 MiB into a freshly allocated output, which is mandatory, since
returning a reused buffer is the aliasing defect above; a fresh 64 MiB CPU tensor costs ~4 ms in
mmap/page-fault zero-fill (6.6 ms fresh plus 2 copies against 2.7 ms into a reused buffer) plus DRAM
bandwidth the NIC's RDMA DMA competes for. The CPU-tensor bench is therefore the worst case for
`all_gather`, whose host passes are 2x `all_reduce`'s per byte; on the GPU path the finish copies
are H2D/D2D DMAs and the fault storm does not exist.

Deliberately not pipelined: the ring `all_reduce` (world > 2 only, unmeasurable on a two-rig fleet)
and `broadcast`; both inherit the cheaper post/wait path but still make unoverlapped host passes.
Deliberately rejected: posting the send directly from a CPU input, and fusing the last accumulate
into the output — both fire only for fp32 CPU tensors, so they would tune the harness, not the
bf16/GPU model path. A C/Cython extension was skipped against a >2x bar: it could recover roughly
the ~5 us of Python around the two posts and the polls on a 27 us collective, an honest ceiling of
~1.3x, and would add a build step on every host, which the ctypes design exists to avoid. At 32 MiB
a chunk-size sweep gave 17.72 Gbit/s at `CHUNK_MIB=2` and 17.34 at 8 against 17.41 at the default 4,
within ~1.5x the run-to-run spread, so the default was left alone.

**Async collectives.** `all_reduce_async` / `all_gather_async` return a handle; `wait_async(handle)`
returns the output. Three contracts, each falsified first:

| Contract | Mechanism |
|---|---|
| Ownership | staging comes from a power-of-two size-class free-list pool, acquired at issue, owned by the handle, released only inside `wait_async`. The caller's input is free the moment issue returns; `all_gather`'s own slice is copied out of the staging slot, never re-read from the input; results are freshly allocated, never pool views; double-wait raises. A release-at-issue mutation flips 10+ lifetime tests red |
| Progress | issue ends with one progress pass, pushing eager sends onto the wire; completion happens under the progress loop in `wait_async`; no progress thread, since the worker is `THREAD_MODE_SINGLE` under the transport lock. Eager-sized payloads move in hardware while the caller computes; rendezvous-sized ones degrade toward sync cost but stay correct |
| Order | `_next_seq` counts issues under the lock, handles carry their seq, tags match exactly, so waits may be out of issue order — tested including a sync collective between issue and wait, and mixed outstanding `all_reduce`/`all_gather` |

Transport-level overlap, CPU tensors over the real link: `sync(coll+busy)` against
`issue+busy+wait` hides nothing, -6 to +4 us. At 8 KiB about 22 of 27 us are local software on the
same CPU as the busy loop; only ~5 us of wire is hideable, and the handle/pool overhead of ~4 us
consumes it.

**Consumer-side overlap**, `SGLANG_HTCCL_UCX_OVERLAP=1`, default off. The MLP all-reduce rides the
existing `fuse_mlp_allreduce` seam, the one legal deferral window in a dense decoder chain: layer
N's `down_proj` all-reduce is mathematically movable into layer N+1's `prepare_attn`, where
all-reduce, residual and layernorm happen together; everything else in the chain is a strict
dependency. Three touch points, all no-ops with the flag off:

| Touch point | Behaviour |
|---|---|
| `should_fuse_mlp_allreduce_with_next_layer` | gains a group-uniform gate on env flag and transport class only, with no rank-local state; the structural guards `moe_cp`, `dp+eagle`, `input_scattered`, `SCATTERED` and `is_last_layer` still veto |
| `RowParallelLinear.forward` | issues `all_reduce_async` at the skip point and attaches `(comm, handle)` to the tensor |
| `prepare_attn` fusion branch | checks for the handle **first**, since a kernel fusion on unreduced data with a handle in flight would double-reduce and orphan the requests, and falls back to the unchanged sync all-reduce when no handle was attached |

Communicator plumbing: `supports_async` / `all_reduce_async` / `wait_async`, with `supports_async`
shaped like `handles()` — payload-independent and group-uniform. 56 registered tests green.

End-to-end on the TP=4 cross-rig arm (numbers above): the transport work is the end-to-end gain and
the async overlap is neutral within run-to-run spread, matching the dependency analysis, since the
deferral window is only the host-side layer boundary. A KV split favouring the fastest card — the
2080 Ti's share moved from 26.6% to 12.5% — is also within spread at that benchmark's short
sequences of ~0.2-2k tokens live context, where attention is a small compute share. Per-rank utilization is stable across all arms, with the slowest card more
than 4x as utilized as any other rank; it paces the lock-step group, and the main rig is 87% idle
waiting in lock step.

**Cross-rig GPU/model bring-up — not yet executed.**

| Step | Requirement |
|---|---|
| Preconditions | the same UCX release on both rigs; `/dev/infiniband` present in the ranks' namespace; RoCE port ACTIVE. The development container has neither `/dev/infiniband` nor a RoCE address, so GPU ranks must run where both the cards and the NIC are visible |
| Flags | `SGLANG_HTCCL=1 SGLANG_HTCCL_TRANSPORT=ucx`, `--nnodes 2 --node-rank {0,1}`, `--dist-init-addr <LAN ip>:<port>` — the control plane stays on 1 GbE, only UCX rides RoCE; per rank `UCX_TLS=rc,self,sm`, `UCX_IB_GID_INDEX=3`, `UCX_NET_DEVICES=<port>` |
| `--enforce-eager` required | like `gloo` and `shm`, this transport synchronises with the host inside every collective, so it cannot be inside a CUDA-graph capture. Only `device` is capturable |
| Model geometry | `tp_size <= q/kv` units; Qwen3.5-4B cannot do TP=5. Any model fitting one card per rig works for a TP=2 smoke test |
| Expected first failures, in likelihood order | pinned-buffer registration cost on the first collective, since staging buffers are `pin_memory=True` only when the rank's device is CUDA; a rank-uniformity break if any path issues a collective on one rig and not the other; the per-collective software floor showing up as poor decode tok/s |
| Validation bar | byte-identical output against a solo run on one rig |

**Cross-rig transport, measured (task #204, 2026-07-26).** Two hosts, one rank per host, CPU
tensors, world=2, 10 iterations per cell, median. Three configurations, chosen so that the
*wire* and the *transport stack* can be told apart: `gloo` over the 1 GbE LAN, `gloo` over the
RoCE NIC's IP (TCP on the fast wire), and HTCCL/UCX native RDMA on that same NIC.

| cell | gloo 1 GbE | gloo RoCE (TCP) | **UCX RDMA** | RDMA vs 1 GbE | RDMA vs gloo-on-same-wire |
|---|---|---|---|---|---|
| barrier | 146.63 us | 116.43 us | **8.30 us** | 17.7x | 14.0x |
| all_reduce 8 KiB | 526.05 us | 362.20 us | **44.92 us** | 11.7x | 8.1x |
| all_gather 8 KiB | 365.28 us | 234.98 us | **41.13 us** | 8.9x | 5.7x |
| all_reduce 64 KiB | 1209.84 us | 318.53 us | **55.79 us** | 21.7x | 5.7x |
| all_gather 64 KiB | 909.12 us | 203.58 us | **64.37 us** | 14.1x | 3.2x |
| all_reduce 512 KiB | 5360.59 us | 547.43 us | **237.70 us** | 22.6x | 2.3x |
| all_gather 512 KiB | 5024.35 us | 608.40 us | **250.45 us** | 20.1x | 2.4x |
| all_reduce 4 MiB | 36588.97 us | 2075.18 us | **1717.39 us** | 21.3x | 1.2x |
| all_gather 4 MiB | 37484.93 us | 3142.25 us | **1693.10 us** | 22.1x | 1.9x |

Reading it: **RDMA's win is latency, not bandwidth.** On the *same* wire it is 14x at barrier
and 8x at 8 KiB, but only 1.2x at 4 MiB — at large payloads both stacks are bandwidth-bound near
the link limit (~16-20 Gbit/s realised). The 1 GbE column saturates at 0.92 Gbit/s, i.e. line
rate, which is the sanity check that the harness measures the wire and not itself.

**This supersedes the "78 us" figure** previously quoted for 1 GbE: that was a raw TCP
round-trip taken during network bring-up, not a collective, and comparing it against UCX
collective numbers understated the gap by roughly 2x. The 1 GbE collective barrier is 146.63 us.

The end-to-end TP=4 arm tables (RDMA vs 1 GbE, even vs uneven split) are condensed into the
boot-check table above; method and full analysis in `TASK_103_SPEC_K_POLICY.md` (task #204).

**Upstream:** SGLang/vLLM distributed backends are NCCL/RCCL only, never bridged (no).
llama.cpp/ik_llama.cpp's RPC backend connects heterogeneous backends over TCP (CUDA/Metal/CPU
confirmed, Vulkan/ROCm **unverified**) but is a backend-delegation/pipeline model, not a collective
substituting for NCCL within one TP group, and is explicitly "proof-of-concept... fragile" per its
own README (partial).

<a id="f22"></a>
### 22. fp8 dequant fallback (W8A16)

Serves fp8 checkpoints on GPUs without a native fp8 GEMM via a dequant W8A16 path
(`CompressedTensorsW8A16Fp8`), gated by a functional capability probe rather than a
capability-number comparison, since `torch.cuda.get_device_capability()` reports `(9,0)` for both
Hopper and gfx900. **Cross-checked, GPU-validated cross-vendor — not merged.** The CUDA path is
unchanged by construction.

| Check | Result |
|---|---|
| Qwen3.5-4B-FP8-dynamic: solo Vega 64, solo 2080 Ti, mixed TP=2 uneven 3,1, mixed TP=2 even 2/2 | all byte-identical, with the solo runs as oracle; neither card is in the sm80-88 range the fp8@3080 caveat covers |
| A separate fused dequant-GEMV kernel for the same lane | decodes raw fp8-e4m3 bytes bit-exact against `torch.to(float32)`, max diff 0.0; mean relative error 0.0014 against 0.0133 for the materialize-then-`F.linear` path it would replace. Not merged, not wired into a model boot; only a pre-merge semantic desk-check exists |
| Open | the non-compressed-tensors `fp8.py` `Fp8Config` family is not wired to the probe |

**Upstream:** sglang requires a native GEMM or Marlin (sm80+); no dequant fallback.

<a id="f23"></a>
### 23. Turing/gfx900 without sgl-kernel

Lets the server start on GPUs the `sgl-kernel` cubin-only wheel does not cover (floor sm80), via a
two-level capability predicate (`sgl_kernel_importable()` / `sgl_kernel_runnable()`) instead of
platform checks, with real fallbacks (`forward_native`, torch-native sampler backend).
**Cross-checked, GPU-validated on both vendors.**

| Check | Result |
|---|---|
| RTX 2080 Ti with `sgl_kernel` absent | all 11 core modules import, server starts, coherent generation, 608 unit tests pass |
| `forward_native` sm75 against gfx900 | byte-identical; against the kernel path it differs by a ~4.8e-07-class reduction-order difference |
| Mixed-vendor TP=2, Triton, HTCCL/`gloo` | reproduces the token ids of both solo runs, with solo as the independent oracle on each vendor |
| Fixes | Turing support; rope/`clamp_position` routing; a 4th platform-vs-availability defect on the NVIDIA rank, reproduced on hardware before the fix (red-then-green) |
| Scope | gfx900 Triton support depends on an external `triton-gcn5` fork, not on fork code |

**Upstream:** no capability-fallback path in sglang for `sgl-kernel`-class dependencies.

<a id="f24"></a>
### 24. SWA-DCP

Token-shards the ~10 global full-attention layers of an SWA-hybrid (Gemma-4 class) by the weighted
owner rule; the ~50 sliding-window layers keep their unsharded local path, so no
`(owner slice ∩ window)` masking arises at all. The window-sharding alternative was measured against
and rejected. **Cross-checked**, merged together with a Gemma-4 text-only mask fix, which is its
required partner — either alone leaves the boot red.

| Item | Detail |
|---|---|
| Requires | `--swa-pool-sizing cap`; in ratio mode the unsharded SWA pool would be scaled by the *global* context budget |
| Refused | HiCache, speculative decoding, MLA, weightless-KV, pure-SWA models |
| Evidence | boots green; a needle planted ~3k tokens beyond the 1024-token sliding window retrieves byte-identical to a TP=1 solo-5090 oracle |
| Throughput | none taken. The ~+6-10% figure in the design note is an ex-ante estimate, not a measurement |
| Carried fix | `_plan_aware_dcp_group_q_head_counts` took `max()` over a hybrid model's two kv-head bases, which is right for a workspace size and wrong for a collective's per-rank counts: for 32 q heads over bases {16, 8} and ratios [5,3,2] the max is `[16,10,8]`, sum 34 against a total of 32. Collectives now use the full-attention base with an exhaustiveness check; single-base models are byte-identical |

Recipe: `docs_new/swa_dcp_stage_b_triton.md` §8.

**Upstream:** no equivalent in sglang.

---

## Guarded / descoped

Built and evaluated, then gated off. No llama.cpp/ik_llama.cpp comparison applies: these concern the
fork's own uneven-TP/DCP machinery, which has no upstream analog.

| Item | Status |
|---|---|
| Tree speculative decoding at `--speculative-eagle-topk > 1` under asymmetric-weighted DCP | Built and GPU-tested; found silently non-greedy under weighted DCP and perf-negative on this rig; restored as a hard fail-fast guard with a CPU test, reproduced on hardware before the guard (red-then-green) |
| Replicated-KV eligibility widened to `kv == tp`, the `<` → `<=` flip | Built, red/green-tested on CPU, GPU-measured; the measurement **refuted** it — at `kv == tp` the alignment repair that makes uneven splits work at `kv < tp` has no room to operate, so it dies on the first forward. Existing `<` semantics kept, with the measured rationale pinned in a test. A genuinely uneven `kv == tp` would need a ragged kernel supporting per-rank non-uniform GQA mapping, not a threshold change |

## Cross-vendor bring-up: additional defects and non-defects

Each is a genuine, independently triggerable defect class found during the cross-vendor campaign.
These are SGLang/Triton-backend-internal — llama.cpp and ik_llama.cpp use a different compute stack,
so no comparison column applies.

| Finding | Detail |
|---|---|
| Even-DCP under the Triton backend silently corrupts output | when KV heads are not replicated across the DCP group; correct only when `tp_size // total_kv_heads >= dcp_size`, the geometry of upstream [sglang #25090](https://github.com/sgl-project/sglang/issues/25090). Now rejected at backend construction; the fork's uneven-DCP geometry is exempt. Left open: uneven DCP combined with the dense-model class, which also produces mojibake, and flashinfer's silent no-op on plain `--dcp-size N` |
| `o_proj` reject-guard for uneven-TP-unaware attention classes | avoids trading a shape error for silent wrong numerics on 3 model classes; folded into row 1 |
| `--rank-kv-ratio` arg gate silently ignoring an unusable token vector | now a hard reject; folded into row 2 |
| `GraphSharedOutput` — suspected shared-buffer defect, confirmed **not** a defect | a falsifier, an unshared and correctly-sized variant, reproduces the shared run bit-for-bit; the buffer is obtained once per runner in `__init__`, not per call, so nothing could hand it to a live consumer. Verified for the overlap scheduler on, which is the shipped configuration; `return_logprob`'s separate read path was not exercised |
| Cold JIT builds collide with the device-collective deadline | `jit_kernel` modules build on first call, and several first calls land in the pre-capture warmup forward, so ranks reach that forward minutes apart on an empty cache. The device transport's wait kernels compare `clock64()` against `_TIMEOUT_CYCLES = 60e9` (~23 s at 2.6 GHz) and trap on expiry, poisoning the CUDA context; the `cudaErrorLaunchFailure` then surfaces on the next, unrelated kernel. Measured 6/6 boots red on a cold cache and 1/1 green with the same tree once warm, stall 23-30 s (red-then-green). Fixed by a cold-build *window* around the warmup forwards rather than by raising the constant, so the deadline baked into the captured graph is unchanged; opened rank-uniformly and unconditionally, since a rank-local predicate in front of a group collective is the hang family that already produced the pynccl and CustomAllreduce defects |
| The JIT kernel cache does not self-heal | a build killed mid-flight leaves `build.ninja` + `cuda.cu` + `cuda_0.o.d` and no `.so`; every later process then dies with `Check failed: (lib_handle_ != nullptr)`. Four such directories accumulated on one host and had to be removed by hand — one interrupted boot turns into a permanent failure. `cache_health.py` classifies entries (complete means a `.so` exists, and is never touched) and discards poison, with a host+pid build marker so a co-located rank's in-flight directory is not mistaken for wreckage |
| Validator hygiene | the campaign's output-corruption validator mis-scored a healthy, math-heavy sample as `CORRUPT`; the faulty letter-fraction rule was removed rather than tuned |

## Scope note

This matrix lists only capabilities with landed code. Planned or partially prototyped items — a
host-RAM tiered-KV fabric for the weightless lane, a draft-KV-pool DCP layout, symmetric
cross-vendor CUDA-graph capture (row 21), and the `fp8.py` `Fp8Config` family on non-CUDA-native
hardware (row 22) — are excluded until they land.

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
