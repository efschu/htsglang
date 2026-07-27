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
`docs_new/weightless_kv_role_precision.md`).

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
