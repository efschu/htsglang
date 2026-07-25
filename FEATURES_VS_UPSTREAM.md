# htsglang Fork Features

Comparison as of 2026-07-22, updated 2026-07-25 (twice: engine additions, then a structural
rewrite for GitHub-width readability); upstream SGLang, vLLM, llama.cpp and ik_llama.cpp. Fork
status reflects two branches, checked directly (not against memory or task lists):
`integration/r3-probe` (repo `wt-merge-probe`, pushed to at least `4c90038a78`, local HEAD
`aec1308973`, which is `4c90038a78` plus a follow-on merge of 7 `feat/htccl-gfx900` commits) and
`feat/htccl-gfx900` (repo `wt-htccl`, tip `3cc2fc9da5`, 9 commits total, of which 2 —
`fa5c507476` and `3cc2fc9da5` — are **not yet merged** into `integration/r3-probe`; detail
sections below say so explicitly where it applies. `integration/r2`, the previous baseline, is
superseded by `integration/r3-probe`. (Note: `/spinning/htsglang` is a stale checkout and was not
used as a source for this update.)

This document lists the features carried by the htsglang fork and records, for each, its
implementation status in the fork and whether an equivalent capability is present in upstream
SGLang, upstream vLLM, `llama.cpp`, or `ik_llama.cpp`.

## Status legend

This project is **work in progress**. Every `Fork`/`Fork status` entry in this document records
the local validation state observed on this fork's own rig(s) at the time of writing — 1x RTX
5090 + 2x RTX 3080 for most rows, plus 1x RTX 2080 Ti (sm75) + 1x Radeon RX Vega 64 (gfx900) for
the cross-vendor rows — not a claim of finished code review, exhaustive test coverage, or
upstream-mergeable maturity. `Implemented` means code is merged and has cleared the specific
tests/boots named in that row's detail section, nothing more; it is not shorthand for "done" in
the sense of fully reviewed and tested, and further validation and hardening is expected to
follow. See the `Fork` column definition below for the full token vocabulary (`Implemented` /
`WIP` / `Exp`).

## Document structure

To stay readable on GitHub without horizontal scrolling, this file is split in two layers:

1. **Overview matrix** (below): one narrow row per feature, verdict tokens only
   (`yes` / `partial` / `no` / `n/a` / `unverified`), linked to its detail section.
2. **Detail sections** (`### N. ...`, one per feature): full description, fork status with
   measurements/caveats, and a per-engine breakdown (verdict + mechanism difference + references).

All measurements, caveats, and references that used to live in the wide table cells now live in
the detail sections — nothing was dropped in the rewrite, only reflowed.

**Column definitions (overview matrix)**
- **Fork** — implementation state in `integration/r3-probe`: `Implemented` (merged, with
  registered tests and/or a validated boot — see the Status legend above; not a claim of
  exhaustive review or final test coverage), `WIP` (present but not complete/validated), `Exp`
  (highly experimental, not production-ready). A trailing `*` means the capability instead lives
  only on the not-yet-merged `feat/htccl-gfx900` branch — see the detail section.
- **SGLang / vLLM** — `yes` / `partial` / `no` / `n/a`, expanded in the detail section.
- **llama.cpp / ik_llama.cpp** — same vocabulary. `partial` means a related capability exists
  through a genuinely **different mechanism** — the difference is named in the detail section,
  never left implicit. `unverified` means the check could not be completed with the sources
  available in this pass (stated rather than guessed).

`n/a` in any column indicates the row's specific comparison point doesn't apply to that engine
(e.g. a base capability like GGUF or PD-disaggregation exists there, but not the fork's delta on
top of it — the detail section always says which).

## Overview matrix

| # | Feature | Fork | SGLang | vLLM | llama.cpp | ik_llama.cpp |
|---|---|---|---|---|---|---|
| [1](#f1) | Asymmetric tensor parallelism | Implemented | no | no | partial | partial |
| [2](#f2) | Asymmetric decode context parallelism | Implemented | partial | partial | no | no |
| [3](#f3) | Rank-to-GPU mapping and co-location | Implemented | no | no | no | no |
| [4](#f4) | Solo drafter placement | Implemented | no | yes | partial | partial |
| [5](#f5) | Cross-algorithm drafter routing | WIP | no | no | no | no |
| [6](#f6) | CUDA graph memory aliasing for spec branches | Implemented | partial | partial | no | no |
| [7](#f7) | MoE expert offload + asymmetric TP/DCP | Implemented | partial | partial | partial | partial |
| [8a](#f8a) | Bespoke GGUF adapter framework | Implemented | no | no | n/a | n/a |
| [8b](#f8b) | Qwen3.5/3.6 GGUF | Implemented | no | no | yes | yes |
| [8c](#f8c) | Gemma-4 GGUF | Implemented | no | no | yes | yes |
| [8d](#f8d) | GGUF K-quant compute kernels | Implemented | partial | partial | yes | yes |
| [8e](#f8e) | Asymmetric-TP x GGUF correctness | Implemented | no | no | n/a | n/a |
| [8f](#f8f) | Multimodal and dynamic-quant GGUF | Implemented | partial | partial | yes | partial |
| [9](#f9) | Hibernate checkpoint/restore | Implemented | no | partial | partial | partial |
| [10](#f10) | Measured VRAM budget | Implemented | partial | partial | partial | partial |
| [11](#f11) | Cross-architecture speculative determinism | Implemented | partial | partial | no | no |
| [12](#f12) | Weightless-KV lane | Implemented | no | no | no | no |
| [13](#f13) | Rig dashboard / planner UI | Exp | n/a | n/a | n/a | n/a |
| [14](#f14) | Single-node PD disaggregation | Implemented | yes (base) | yes (base) | no | no |
| [15](#f15) | Asymmetric-TP quantization correctness | Implemented | partial | partial | n/a | n/a |
| [16](#f16) | Fast-lane priority scheduling | Implemented | partial | partial | no | no |
| [17](#f17) | HiCache under asymmetric-TP/DCP | Implemented | yes (base) | n/a | partial | partial |
| [18](#f18) | TP greater than num_kv_heads | Implemented | partial | partial | partial | partial |
| [19](#f19) | Broad model bring-up under asymmetric-TP | Implemented | n/a | n/a | n/a | n/a |
| [20](#f20) | Session KV spill | Exp | partial | partial | partial | partial |
| [21](#f21) | HTCCL cross-vendor collectives | Implemented | no | no | partial | partial |
| [22](#f22) | fp8 dequant fallback (W8A16) | Implemented* | no | no | partial | unverified |
| [23](#f23) | Turing/gfx900 without sgl-kernel | Implemented | no | no | partial | partial |

---

## Detail sections

<a id="f1"></a>
### 1. Asymmetric tensor parallelism

**Feature:** for heterogeneous / mismatched GPUs (`--rank-tp-ratio auto`): unequal per-rank
attention-head / weight shard sizes within a single tensor-parallel group on one node.

**Fork status:** Implemented — validated at TP=3 on 1x RTX 5090 + 2x RTX 3080 (Qwen3.6-27B FP8).
Greedy decode is *self-deterministic*: byte-identical run-to-run and cold-vs-warm at a fixed
config on the same GPUs (not a claim of cross-hardware determinism, see row 11).
**2026-07-25:** an `o_proj`-vs-head-split audit (mechanical sweep of every `RowParallelLinear` in
`python/sglang/srt/models/`) found 3 architectures (`qwen2`, `qwen3`, `qwen3_moe`) whose attention
consults the uneven-TP ratio while their own head count stays computed as an even split — a
silent-shape-agreement / silent-wrong-numerics trap. Fixed by rejecting a non-uniform plan at
construction time on those classes (`dd68fad951`, test
`test_uneven_tp_is_rejected_on_models_whose_attention_is_not_aware`); the other 95 unaware sites
are architectures this fork does not run under uneven TP and are left latent, not fixed. Also
**2026-07-25**: DFLASH gained per-rank attention/MLP shards for uneven TP (`5af72c7a60`, merged
`734f77e313`), validated green together with prefill-spill in the `integration/r3-probe` boot
matrix (arm I: MLP vector `2,1,1`, units `[68,34,34]`). A separate proposal to widen
replicated-KV eligibility to `kv == tp` (`<` -> `<=`) was implemented, GPU-measured, and
**reverted**: it boots but dies on the first forward on a config the alignment machinery cannot
repair at `kv == tp` (`63c06d97a4`,
`test_kv_eq_tp_stays_in_normal_mode_by_measurement` pins the measured reason).

**Engine comparison:**
- **SGLang:** no — TP path requires head counts divisible by TP size; equal shards per rank.
- **vLLM:** no — TP path requires equal per-rank sharding.
- **llama.cpp:** partial — `--tensor-split N0,N1,...` plus `--split-mode {layer,row,tensor}`
  distributes genuinely *unequal* proportions across GPUs, but the split is by whole layer or by
  matrix row, never by attention head, and there is no per-rank-process/broadcast execution model
  to speak of (single process owns all devices). Same honest distinction the coordinator's example
  called out: `partial`, not `yes`.
- **ik_llama.cpp:** partial — inherits the same `-ts`/`-sm` flags; its own docs
  ([parameters.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md))
  list `-sm` values `none`/`graph`/`layer` (row/tensor modes not confirmed present in that
  listing — **unverified** whether they were dropped or just undocumented). Same row/layer
  mechanism gap as llama.cpp.

**References:** [vLLM parallelism scaling docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling)
· [HexGen (2023)](https://arxiv.org/abs/2311.11514) · [Hetis (SC'25)](https://dl.acm.org/doi/10.1145/3712285.3759784)
· [Tangram (2026)](https://arxiv.org/pdf/2606.16907) · [Cronus (2025)](https://arxiv.org/pdf/2509.17357)
· [Tessera (2026)](https://arxiv.org/pdf/2604.10180)
· [llama.cpp `--tensor-split`/`--split-mode`](https://github.com/ggml-org/llama.cpp)
· [ik_llama.cpp parameters](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md)

<a id="f2"></a>
### 2. Asymmetric decode context parallelism

**Feature:** (`--rank-kv-ratio`; env still `SGLANG_UNEVEN_DCP`, aligned in the later rename):
capacity-weighted KV-cache ownership across ranks during decode.

**Fork status:** Implemented — token-split variant validated. **2026-07-25 hardening:** the arg
gate (`--rank-kv-ratio` requiring a non-uniform `--rank-tp-ratio`) was audited on request ("is
this dependency real or removable?") and confirmed to be a genuine wiring dependency in three
places, down to `uneven_dcp_kv_replicated()`'s own definition — not arbitrary.
`resolve_cp_token_ratios` previously **silently ignored** an explicit token vector set without a
plan (measured: boots green, zero uneven-machinery log lines, output identical to plain TP — a
configured-looking server doing nothing that was asked); it now **rejects** that case instead
(`4c90038a78`, test `test_token_vector_without_a_plan_is_rejected_not_ignored`). Also
**2026-07-25**: even (non-fork) `--dcp-size N` under the Triton backend was found to silently
produce mojibake whenever `tp_size // total_kv_heads < dcp_size` (kv heads not replicated across
the DCP group) — root-caused and guarded, see the cross-vendor bugfix list below; the fork's own
uneven-DCP geometry is exempted and unaffected.

**Engine comparison:**
- **SGLang:** partial — DCP present ([issue #12196](https://github.com/sgl-project/sglang/issues/12196),
  roadmap [#21788](https://github.com/sgl-project/sglang/issues/21788)); KV cache split evenly
  (token index mod world size).
- **vLLM:** partial — DCP present via `decode_context_parallel_size` / `-dcp`; symmetric split.
- **llama.cpp:** no — no context-parallel decode of any kind found; the RPC backend distributes
  whole layers/devices across hosts (pipeline-style), not per-token KV ownership.
- **ik_llama.cpp:** no — same, inherits only the RPC layer-distribution model, no KV-split
  context parallelism.

**References:** [SGLang DCP issue](https://github.com/sgl-project/sglang/issues/12196)
· [SGLang DCP roadmap](https://github.com/sgl-project/sglang/issues/21788)
· [vLLM context parallel deployment](https://docs.vllm.ai/en/main/serving/context_parallel_deployment.html)
· [Helix Parallelism (2025)](https://arxiv.org/pdf/2507.07120) · [Medha (2024)](https://arxiv.org/pdf/2409.17264)
· [Context Parallelism for Million-Token Inference (2024)](https://arxiv.org/pdf/2411.01783)

<a id="f3"></a>
### 3. Rank-to-GPU mapping and co-location

**Feature:** (`--rank-gpu-id`, `--rank-gpu-memory-mib`): assigns each rank to a named physical GPU
(NVML-resolved), permitting multiple ranks per GPU.

**Fork status:** Implemented — co-location path requires NCCL >= 2.30 (shipped in the fork's
Docker image). **2026-07-25:** `--rank-tp-ratio` / `--rank-kv-ratio` no longer require
`--rank-gpu-id` to be set (`c51dd9c371`) — sharding-ratio validity and physical placement are
independent concerns, and coupling them concretely blocked the cross-vendor case (row 21), where
placement is two OS launchers on one host, not one NVML-resolved vector, since NVML cannot name
an AMD rank. Validation was hoisted above the placement early-return so it still fires with no
`--rank-gpu-id` (length mismatch / all-identical / non-positive entries all re-verified rejected);
the two genuine mutual requirements (`--rank-gpu-id` <-> `--rank-gpu-memory-mib`) are unchanged.

**Engine comparison:**
- **SGLang:** no — placement via `CUDA_VISIBLE_DEVICES` / device ordinals; no per-rank physical-GPU
  index flag.
- **vLLM:** no — placement via `CUDA_VISIBLE_DEVICES` / device ordinals; no per-rank physical-GPU
  mapping flag.
- **llama.cpp:** no — has no "rank" concept at all: one process owns a device list
  (`--device`/`--tensor-split`/`--main-gpu`), or one `ggml-rpc-server` process exposes an entire
  remote host's devices. There is no per-rank physical-GPU vector, and no co-location primitive.
- **ik_llama.cpp:** no — same single-process device-list model (`-dev`), no rank concept.

**References:** [NVIDIA MPS](https://docs.nvidia.com/deploy/mps) · [CUDA_VISIBLE_DEVICES](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
· [NVML device identity](https://docs.nvidia.com/deploy/nvml-api)
· [llama.cpp RPC backend](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc)

<a id="f4"></a>
### 4. Solo drafter placement

**Feature:** (`--speculative-draft-placement solo`): runs the draft model unsharded on one GPU and
broadcasts its output instead of all-reduce.

**Fork status:** Implemented — with registered unit tests (solo placement, weight/KV planning,
vocab broadcast).

**Engine comparison:**
- **SGLang:** no — EAGLE/EAGLE3 draft runs under the model's TP layout (`--speculative-algorithm`,
  `--speculative-draft-model-path`); no `speculative-draft-tensor-parallel-size` flag.
- **vLLM:** yes — `--speculative-draft-tensor-parallel-size 1` runs an unsharded draft with a
  sharded target.
- **llama.cpp:** partial — `--spec-draft-device`, `--spec-draft-ngl`, `--spec-draft-override-tensor`
  pin the draft model to its own device(s) independent of the target's `--tensor-split`, reaching
  the same *outcome* (draft unsharded on one GPU while the target is split). But there is no
  all-reduce/broadcast primitive at all in llama.cpp's single-process compute-graph model — the
  mechanism is plain per-model device assignment, not a TP-rank broadcast substituting for an
  all-reduce, so the row's specific claim (broadcast-instead-of-all-reduce) doesn't transfer.
- **ik_llama.cpp:** partial — same idea via `-devd`/`--device-draft`, `-ngld`/`--gpu-layers-draft`
  per its [parameters doc](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md);
  a narrower flag set than llama.cpp's (no `--override-tensor-draft` confirmed — **unverified**
  whether it's simply undocumented). Same mechanism gap as llama.cpp.

**References:** [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
· [Disaggregated Standalone Draft RFC (2026)](https://github.com/vllm-project/vllm/issues/42109)
· [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
· [EasySpec (2025)](https://arxiv.org/pdf/2502.02493)
· [llama.cpp `--spec-draft-*` device flags](https://github.com/ggml-org/llama.cpp)

<a id="f5"></a>
### 5. Cross-algorithm drafter routing

**Feature:** (`--speculative-cross-algorithm*`, `SGLANG_CROSS_*` — the CLI/env name for this
"drafter routing"): two draft algorithms (NEXTN/MTP and DFLASH) resident simultaneously, selected
per batch at round boundaries by a bandit controller (accept-tokens/round / round duration,
rank-0 decision + TP broadcast), with a context-size-aware length gate from the drafter training
config.

**Fork status:** Work in progress — dual residence, per-batch switch, and the bandit controller
are implemented (`cross_algo_worker`, with a registered bandit test); the context-length gate from
the drafter training config is not yet implemented. **2026-07-25 addition:** lazy single-graph
capture + DFLASH context-retirement (#156-4) merged (`f2c96f31b3`) — validated green under CUDA
graphs + full three-axis programme (`integration/r3-probe` arm C: lazy capture ACTIVE, adaptive
graph memory released 542.0 MiB; arm G: cross-algo + HTCCL + offload + spec-in-tick together,
green).

**Engine comparison:**
- **SGLang:** no — adaptive speculative decoding adjusts `speculative_num_steps` /
  `num_draft_tokens` for one drafter; no switching between draft algorithms.
- **vLLM:** no — per-request `k` adaptation, confidence early-exit, disable-by-batch-size; single
  drafter, no algorithm switching.
- **llama.cpp:** no — one draft model/algorithm at a time (`--model-draft`, plus lookahead and
  prompt-lookup n-gram drafting as *alternative* strategies, not co-resident with runtime
  switching); no bandit controller between resident algorithms.
- **ik_llama.cpp:** no — same single-drafter model inherited from llama.cpp.

**References:** [SGLang adaptive spec decoding](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding)
· [SGLang roadmap #23705](https://github.com/sgl-project/sglang/issues/23705)
· [AutoSpec RFC](https://github.com/sgl-project/sglang/issues/15319) · [Dynamic SPD](https://github.com/sgl-project/sglang/issues/9319)
· [DSpark blog](https://lmsys.org/blog/2026-07-06-dspark-sglang) · [vLLM DynamicProposer](https://github.com/vllm-project/vllm/pull/26504)
· [vLLM DSL RFC](https://github.com/vllm-project/vllm/issues/36657) · [vLLM Automate Spec RFC](https://github.com/vllm-project/vllm/issues/4565)
· [BanditSpec (ICML 2025)](https://arxiv.org/abs/2505.15141) · [OnlineSpec (ICML 2026)](https://arxiv.org/abs/2603.12617)
· [LongSpec (2025)](https://arxiv.org/abs/2502.17421)

<a id="f6"></a>
### 6. CUDA graph memory aliasing for spec branches

**Feature:** (#93/#102): inactive speculative-depth CUDA-graph branches hold no physical VRAM via
cuMem tag aliasing.

**Fork status:** Implemented — `kv_vmm_backing` / adaptive runtime state.

**Engine comparison:**
- **SGLang:** partial — pre-capturing multiple speculative CUDA graphs is on the roadmap
  ([#23705](https://github.com/sgl-project/sglang/issues/23705)); physical release of inactive
  branch allocations via VMM aliasing not documented.
- **vLLM:** partial — cuMem VMM allocator with tag-based offload exists (Sleep Mode); not applied
  to speculative CUDA-graph branches.
- **llama.cpp:** no — ggml-cuda has CUDA graph capture/replay for repeated decode shapes
  (`USE_CUDA_GRAPH` in `ggml-cuda.cu`), but speculative decoding there is a single chain/tree, not
  a set of alternate depth-indexed graphs, so there is nothing analogous to alias.
- **ik_llama.cpp:** no — a code search for `cuda_graph` in the project turned up no hits; CUDA
  graph capture does not appear to be implemented there at all, let alone the aliasing refinement.

**References:** [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
· [vLLM CuMemAllocator](https://docs.vllm.ai/en/latest/api/vllm/device_allocator/cumem/)
· [SGLang multi-spec-graph roadmap](https://github.com/sgl-project/sglang/issues/23705)
· [CUDA Virtual Memory Management API](https://docs.nvidia.com/cuda/cuda-driver-api)

<a id="f7"></a>
### 7. MoE expert offload + asymmetric TP/DCP

**Feature:** MoE expert offloading to host RAM combined with asymmetric TP and asymmetric DCP
(GPTQ/AWQ/FP8).

**Fork status:** Implemented — validated on a 122B A10B MoE across three mismatched GPUs.

**Engine comparison:**
- **SGLang:** partial — `--cpu-offload-gb` layer-granular weight offload
  ([PR #3675](https://github.com/sgl-project/sglang/pull/3675), `offload_group_size`);
  expert-granular offload is an open feature request
  ([#14233](https://github.com/sgl-project/sglang/issues/14233)); not combined with asymmetric
  TP/DCP.
- **vLLM:** partial — `--cpu-offload-gb` general weight offload via UVA; not expert-granular; not
  combined with asymmetric TP/DCP.
- **llama.cpp:** partial — `-ot`/`--override-tensor` (regex-based per-tensor placement) plus
  `-ncmoe`/`--n-cpu-moe` (keep the first N layers' MoE weights on CPU) is the same expert-granular
  host-offload idea, but there is no asymmetric-TP/DCP to combine it with, since neither concept
  exists in llama.cpp.
- **ik_llama.cpp:** partial, and worth calling out — same `-ot`/`--cpu-moe`/`-n-cpu-moe` flags, and
  this project is widely credited in the community as an early driver of practical MoE CPU+GPU
  hybrid offload for models like DeepSeek-V3 (row-interleaved quant kernels tuned specifically for
  that CPU/GPU split). Still no asymmetric-TP/DCP to compose it with.

**References:** [vLLM offload config](https://docs.vllm.ai/en/latest/api/vllm/config/offload/)
· [SGLang offload PR](https://github.com/sgl-project/sglang/pull/3675) · [SGLang expert-granular request](https://github.com/sgl-project/sglang/issues/14233)
· [KTransformers](https://github.com/kvcache-ai/ktransformers) · [llama.cpp `--override-tensor`](https://github.com/ggml-org/llama.cpp)
· [MoE-Infinity (2024)](https://arxiv.org/html/2401.14361) · [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

<a id="f8a"></a>
### 8a. Bespoke GGUF adapter framework

**Feature:** (#129): a registry (`gguf_registry`) + `GGUFAdapterBase` that add per-model-family
GGUF loaders (name maps + llama.cpp inverse weight transforms) on top of the generic GGUF path,
and read `config.json` / `tokenizer.json` from sibling files for archs whose GGUF metadata
`transformers` cannot parse.

**Fork status:** Implemented — registry with two families; unit tests (header, sizing).

**Engine comparison:**
- **SGLang:** no — generic GGUF path cannot load these hybrid/new arches (metadata reader
  crashes; name map differs).
- **vLLM:** no — generic GGUF path only.
- **llama.cpp:** n/a — GGUF is llama.cpp's native, first-party format. There is no "adapter" layer
  to compare against because the reference converter/loader (`convert_hf_to_gguf.py`,
  `src/models/*.cpp`) **is** the direct implementation, not a compatibility shim bolted onto a
  generic reader. This row's comparison point (adapter-over-generic-path) structurally doesn't
  apply to the format's own home.
- **ik_llama.cpp:** n/a — same reasoning; it carries its own `src/llama.cpp` with native arch code
  (confirmed present via a `GEMMA4`/`QWEN35`/`QWEN3NEXT` enum grep on `main`, 2026-07-25), not a
  bolt-on adapter.

**References:** [vLLM GGUF docs](https://docs.vllm.ai/en/stable/features/quantization/gguf/)
· [llama.cpp GGUF + gguf-py tensor maps](https://github.com/ggml-org/llama.cpp)
· [GGUF format spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

<a id="f8b"></a>
### 8b. Qwen3.5/3.6 GGUF

**Feature:** GGUF arch `qwen35` dense hybrid-GDN and `qwen35moe`: inverse transforms (GDN
value-head retiling, RMSNorm `+1` un-bake, `out_proj` byte-vs-element un-tile, de-fused GDN
`in_proj_qkvz`), plus NEXTN/MTP draft loaded from the same file (`blk.<num_layers>`), including
MoE draft with routed experts.

**Fork status:** Implemented — dense + MoE + NEXTN/MTP; K-quants Q4_K_M ... Q8_0 coherent and
greedy-deterministic; validated Q6_K, asymmetric TP=3 (5090 + 2x 3080).

**Engine comparison:**
- **SGLang:** no — arch unsupported by the generic path.
- **vLLM:** no — arch unsupported by the generic path.
- **llama.cpp:** yes — native arch support confirmed directly in the local checkout
  (`/spinning/llm_stuff/llama.cpp-master`, commit `0c4fa7a989`, 2026-07-12):
  `src/models/qwen35.cpp`, `qwen35moe.cpp`, `qwen3next.cpp` all exist, plus
  `SSM_BETA_ALPHA # qwen3next` in `gguf-py/gguf/constants.py`. llama.cpp is the format's home here
  — it is ahead of, not behind, this fork's inverse-transform port.
- **ik_llama.cpp:** yes, with a caveat — `QWEN35`/`QWEN3NEXT` enum constants are present in
  `src/llama.cpp` on `main` (checked via raw GitHub fetch, 2026-07-25), so the architecture exists
  in source. Not independently GPU-verified here whether NEXTN/MTP draft-from-same-file loading
  specifically works end-to-end on that fork.

**References:** [Qwen3.5 GGUF evals + speculative (2026)](https://kaitchup.substack.com/p/more-qwen35-gguf-evals-and-speculative)
· [SGLang speculative decoding docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
· [llama.cpp `src/models/qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)

<a id="f8c"></a>
### 8c. Gemma-4 GGUF

**Feature:** GGUF arch `gemma4`, dense: inverse transforms distinct from Qwen (quantized
`token_embd` dequantized to a dense embedding; **identity** norm handling — the Gemma export does
not bake `+1` into RMSNorm gammas, the opposite of the Qwen export; tied `lm_head`; `k==v` shard
duplication on full-attention layers).

**Fork status:** Implemented — Gemma-4-31B-it Q4_K_M validated (TP=1 on RTX 5090, ~61 tok/s,
coherent + self-deterministic; asymmetric TP=3 green). MoE / MTP / vision Gemma-4 GGUF deferred
(fail-fast). Only Q4_K_M empirically verified.

**Engine comparison:**
- **SGLang:** no — arch unsupported by the generic path (metadata reader derives a per-layer
  `num_key_value_heads` list the strict config rejects).
- **vLLM:** no — arch unsupported by the generic path.
- **llama.cpp:** yes — native (`src/models/gemma4.cpp`, `gemma4-assistant.cpp`;
  `GEMMA4`/`GEMMA4_ASSISTANT` constants confirmed in the local checkout).
- **ik_llama.cpp:** yes, with a caveat — `GEMMA4`/`gemma4` present in `src/llama.cpp` on `main`
  (2026-07-25 fetch); not independently verified here for MoE/vision/MTP sub-variants.

**References:** [Gemma docs](https://ai.google.dev/gemma) (Gemma requires `--attention-backend triton`, rejects flashinfer)
· [llama.cpp `src/models/gemma4.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp)

<a id="f8d"></a>
### 8d. GGUF K-quant compute kernels

**Feature:** (`sgl-kernel` MMQ/MMVQ): tuned K-quant MMVQ (K-split + column amortization), a
per-device MMVQ<->MMQ crossover (MMQ capped to small batch — the stock path used MMQ for K-quants
at all batch sizes, penalizing prefill), batched MMVQ (`ncols_dst <= 8`), quantized
vocab/embedding, Q8_0, and I-Matrix quant via MMVQ.

**Fork status:** Implemented — merged from `feat/kquant-kernel`; kernel tests. **2026-07-25
(#163, `a595fb493a`, merged `0bc9e068c6`):** opt-in `--gguf-mmq-decode-threshold`, default OFF,
reroutes decode from MMVQ to MMQ above a *measured, per-device, per-shape-class* bucket (sm120: 8
for every shape class; sm86: none — MMVQ still wins there at M=8). Coupled to the CUDA-graph
decode bucket so a replay never runs a different kernel than it was captured with (a
registration-ordering defect in the same merge was caught and fixed: bucket registration must run
*after* the last pass that shrinks `capture_bs`,
`test_decode_bucket_registration_happens_after_the_last_capture_bs_edit`). Measured gain,
Qwen3.6-27B UD-Q8_K_XL, TP=3 (1x5090+2x3080), CUDA graphs, three content classes: **+9.7-10.6%**
end-to-end tok/s, but **only the sm120 rank reroutes** (confirmed by per-rank kernel-call counts:
11320 MMQ / 0 MMVQ on TP0, 0 MMQ / 11320 MMVQ on TP1/TP2) — 2 of 3 ranks on this rig see no
change. **Not byte-identical when ON**: greedy token ids diverge from the flag-off run (2/8, 2/8,
4/8 matching heads across three prompts) because MMQ and MMVQ reduce in a different order; flag
OFF reproduces the pre-existing dispatch exactly (0 threshold-engagement log lines).

**Engine comparison:**
- **SGLang:** partial — generic GGUF MMQ/MMVQ kernels exist; the per-device crossover,
  prefill-oriented MMQ cap, and quantized-vocab path are fork additions.
- **vLLM:** partial — generic GGUF MMQ/MMVQ kernels exist; same additions not present.
- **llama.cpp:** yes (base) — MMQ/MMVQ kernels originate here (`ggml-cuda`); the fork's specific
  per-device crossover threshold, prefill-oriented cap, and quantized-vocab tuning are the fork's
  own addition on top of llama.cpp's kernels, not something llama.cpp itself does.
- **ik_llama.cpp:** yes (base), and independently notable — this project maintains its own
  separate, actively-optimized kernel lineage (`iqk_mul_mat`, row-interleaved quant layouts) rather
  than just inheriting llama.cpp's MMQ/MMVQ, so it is a genuinely parallel optimization effort, not
  a downstream copy. The fork's specific per-device crossover heuristic is still fork-only.

**References:** [vLLM GGUF quant kernels](https://docs.vllm.ai/en/stable/features/quantization/gguf/)
· [llama.cpp MMQ/MMVQ](https://github.com/ggml-org/llama.cpp)
· [ik_llama.cpp key features (DeepWiki)](https://deepwiki.com/ikawrakow/ik_llama.cpp/1.1-key-features-and-performance-improvements)

<a id="f8e"></a>
### 8e. Asymmetric-TP x GGUF correctness

**Feature:** composes GGUF with feature #1: K-quant 256-element superblock alignment on every
per-rank column/row split; GDN `in_proj` head-partition block coarsening; `Qwen2MoeMLP`
vec-aligned, quant-symmetric TP units; GGUF-MoE MMQ out-of-bounds expert-id fixes under
asymmetric expert sharding; per-rank local-expert-count guard. Same group-alignment applied to
compressed-tensors AWQ/GPTQ INT4.

**Fork status:** Implemented — a series of merged bugfixes (#80, #81, #82, #109) with registered
tests.

**Engine comparison:**
- **SGLang:** no — asymmetric TP itself is absent upstream, so these alignment paths do not
  exist.
- **vLLM:** no — asymmetric TP absent upstream.
- **llama.cpp:** n/a — the closest analog is `--split-mode row`, which does split individual
  quantized tensors by row across GPUs; being the reference K-quant implementation, that path is
  presumably correct by construction for its own splitting granularity. But true per-rank
  asymmetric TP (independent uneven attention-head shards, the thing this row's bugfixes are
  about) does not exist there, so the bugfix class itself doesn't arise. **Unverified**: whether
  `--split-mode row` at an uneven GPU *count* (not an uneven *ratio*, which it doesn't expose) ever
  hits a K-quant superblock boundary issue — not found in the time available.
- **ik_llama.cpp:** n/a — same reasoning; asymmetric TP is absent there too.

**References:** depends on feature #1 (asymmetric TP)
· [llama.cpp/ggml K-quant block layout](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

<a id="f8f"></a>
### 8f. Multimodal and dynamic-quant GGUF

**Feature:** load a vision tower from a companion `mmproj` GGUF; load unsloth "UD" dynamic-quant
GGUFs (dense-F16 `out_proj`, mixed precision).

**Fork status:** Implemented — UD Q6_K_XL (+ mmproj vision tower) validated in the benchmark
matrix; UD Q8_K_XL infeasible on the reference rig (size + a known Q8 loader limitation).

**Engine comparison:**
- **SGLang:** partial — generic GGUF path does not load these companion-`mmproj` /
  mixed-precision UD variants for the affected arches.
- **vLLM:** partial — same.
- **llama.cpp:** yes — `mmproj` companion vision-tower loading and unsloth "UD" dynamic-quant
  GGUFs load through the generic/native path already; this is the format's home implementation,
  and unsloth quantizes specifically for llama.cpp compatibility.
- **ik_llama.cpp:** partial — community reports indicate multimodal/`mmproj` support lags
  mainline llama.cpp on this fork; UD dynamic-quant GGUFs load fine wherever the underlying
  architecture itself is supported, so the verdict is architecture-dependent.
  **Unverified in detail** — flagged as partial rather than a confident yes or no.

**References:** [unsloth dynamic GGUF quants](https://huggingface.co/unsloth)
· [vLLM/llama.cpp GGUF docs](https://docs.vllm.ai/en/stable/features/quantization/gguf/)

<a id="f9"></a>
### 9. Hibernate checkpoint/restore

**Feature:** (#89): persists warm server state to disk so it survives process exit and reloads
without full initialization.

**Fork status:** Implemented and validated for dense GGUF (load 50s -> 8-14s under asymmetric
TP=3, survives process exit; `model_loader/hibernate`); the FP8 path is functional with
negligible load-time benefit; MoE-model hibernation deferred.

**Engine comparison:**
- **SGLang:** no — offload/wake-up exists for the diffusion server
  ([PR #19152](https://github.com/sgl-project/sglang/pull/19152)); no full-state disk snapshot for
  the LLM server.
- **vLLM:** partial — Sleep Mode releases/restores memory in-process; CUDA checkpoint/restore to
  persistent snapshot proposed ([RFC #34303](https://github.com/vllm-project/vllm/issues/34303)),
  not merged.
- **llama.cpp:** partial — `--prompt-cache`/`--prompt-cache-ro` (KV-state file) and the server's
  `/slots/{id}?action=save|restore` endpoint persist per-conversation KV state to disk and reload
  it. This is prompt/session-level state, not a full-process hibernate: weights and allocator
  state are not snapshotted, the model is reloaded fresh and only the KV/prompt state comes back.
- **ik_llama.cpp:** partial — same slot save/restore endpoints present
  (`SERVER_TASK_TYPE_SLOT_SAVE`, `params.slot_save_path` confirmed in `examples/server/server.cpp`
  on `main`), same scope limitation as llama.cpp.

**References:** [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
· [vLLM CUDA C/R RFC](https://github.com/vllm-project/vllm/issues/34303)
· [SGLang offload/wake-up PR](https://github.com/sgl-project/sglang/pull/19152)
· [ServerlessLLM (2024)](https://arxiv.org/pdf/2401.14351) · [Tangram (2025)](https://arxiv.org/pdf/2512.01357)
· [PipeBoost (2025)](https://arxiv.org/pdf/2503.17707) · [CRIU](https://criu.org)
· [llama.cpp `--prompt-cache`](https://github.com/ggml-org/llama.cpp) · [llama.cpp server slot save/restore](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

<a id="f10"></a>
### 10. Measured VRAM budget

**Feature:** (component registry, measured KV-cache remainder, two-boot convergence,
`--rank-gpu-memory-mib`): per-rank absolute MiB budget derived from measured component usage
rather than a global fraction.

**Fork status:** Implemented — per-rank absolute MiB budget plus self-calibrating KV split (boot
logs a vector hint fed back on restart).

**Engine comparison:**
- **SGLang:** partial — memory profiling with fraction-based `mem-fraction-static`; per-rank
  absolute MiB budget not present.
- **vLLM:** partial — memory profiling with fraction-based `gpu-memory-utilization`; per-rank
  absolute MiB budget not present.
- **llama.cpp:** partial — `-fit`/`--fit` auto-adjusts unset ngl/ctx/batch parameters to fit a
  **declared free-memory target** (`--fit-params-target`, MiB, one value broadcast to all devices
  or a per-device list), with `--fit-print` to show the estimate. Conceptually close (absolute-MiB,
  per-device budget), but it sizes model/context parameters to fit a target rather than deriving a
  per-rank *fraction* from measured component usage with a two-boot self-calibrating KV vector.
- **ik_llama.cpp:** partial — recent project updates add "auto-fit offloaded tensors to available
  VRAM" for both MoE and dense models (per project docs/DeepWiki, 2026); same spirit as llama.cpp's
  `-fit`, not independently confirmed to match the mechanism in full detail.

**References:** [vLLM memory config](https://docs.vllm.ai/en/stable/configuration/optimization/)
· [vLLM offload config](https://docs.vllm.ai/en/latest/api/vllm/config/offload/)
· [SGLang `mem-fraction-static`](https://docs.sglang.io) · [NVML memory query](https://docs.nvidia.com/deploy/nvml-api)
· [llama.cpp `-fit`/`--fit-params-target`](https://github.com/ggml-org/llama.cpp)
· [ik_llama.cpp key features (DeepWiki)](https://deepwiki.com/ikawrakow/ik_llama.cpp/1.1-key-features-and-performance-improvements)

<a id="f11"></a>
### 11. Cross-architecture speculative determinism

**Feature:** verify synchronization and CUDA-graph padding across sm86 + sm120; sampling
broadcast from rank 0.

**Fork status:** Implemented — the three divergence root causes resolved; the emitted greedy
*token sequence* is reproducible across the mixed-architecture TP group. This is
output-preserving reproducibility (spec decode emits the target model's argmax chain, and the
accept decision is taken once on rank 0 and broadcast), **not** bit-identical activations — sm86
and sm120 reduce in different order, so the low-order bits differ and are not claimed to match.

**Engine comparison:**
- **SGLang:** partial — deterministic inference mode exists
  (`--enable-deterministic-inference`, batch-invariant ops); mixed-GPU-architecture TP groups not
  addressed.
- **vLLM:** partial — determinism work exists (batch-invariant ops); mixed-GPU-architecture TP
  groups not addressed.
- **llama.cpp:** no — no mixed-vendor/mixed-architecture TP determinism engineering found. The
  RPC backend could in principle connect heterogeneous backends, but no verify-sync/graph-pad/
  workspace-alignment work analogous to the fork's three root causes is documented anywhere found.
- **ik_llama.cpp:** no — same, no evidence found.

**References:** [SGLang deterministic inference docs](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/deterministic_inference.md)
· [SGLang determinism issue](https://github.com/sgl-project/sglang/issues/10278) · [SGLang determinism blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic)
· [Defeating Nondeterminism in LLM Inference (Thinking Machines)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
· [Batch Speculative Decoding Done Right (2025)](https://arxiv.org/pdf/2510.22876) · [Bit-Exact AI Inference Verification (2026)](https://arxiv.org/pdf/2606.00279)

<a id="f12"></a>
### 12. Weightless-KV lane

**Feature:** (`--weightless-kv-fastlane`; the `fastlane` in the flag name is historical and is
**unrelated** to the separate fast-lane priority scheduling of row 16 — read this as the
*weightless-KV lane*, the weight-bearing side being the *head lane*): a meta-device worker holds
only KV cache and attention while the weight-bearing head holds the weights.

**Fork status:** Implemented — chunked prefill and graph-decode paths in place.

**Engine comparison:**
- **SGLang:** no.
- **vLLM:** no.
- **llama.cpp:** no.
- **ik_llama.cpp:** no.

**References:** [Adrenaline (2025)](https://arxiv.org/pdf/2503.20552) · [Mooncake (FAST'25)](https://www.usenix.org/system/files/fast25-qin.pdf)
· [CrossPool (2026)](https://arxiv.org/pdf/2606.24506)

<a id="f13"></a>
### 13. Rig dashboard / planner UI

**Feature:** capacity-planning tool reporting work-normalized J/token under asymmetric DCP.

**Fork status:** Highly experimental — functional but under active development, not
production-ready (`tools/rig_dashboard`).

**Engine comparison:**
- **SGLang:** n/a — external tooling; SGLang exposes Prometheus metrics.
- **vLLM:** n/a — external tooling; vLLM exposes Prometheus metrics
  ([design doc](https://docs.vllm.ai/en/latest/design/metrics/)).
- **llama.cpp:** n/a — same category; llama.cpp exposes its own Prometheus-compatible `/metrics`
  endpoint (`--metrics`, `tools/server/README.md`) plus `llama-bench` for offline benchmarking.
- **ik_llama.cpp:** n/a — same category; inherited metrics/bench tooling from the llama.cpp fork
  point.

**References:** [NVIDIA DCGM](https://developer.nvidia.com/dcgm) · [NVML](https://docs.nvidia.com/deploy/nvml-api) · [Grafana](https://grafana.com)
· [vLLM metrics design](https://docs.vllm.ai/en/latest/design/metrics/) · [SGLang metrics docs](https://docs.sglang.io)
· [llama.cpp server `--metrics`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

<a id="f14"></a>
### 14. Single-node PD disaggregation

**Feature:** prefill/decode disaggregation, single-node heterogeneous split (`local_proxy`,
`mooncake_tcp` loopback): prefill runs solo on the fastest card (TP=1, no cross-GPU traffic),
decode runs distributed under asymmetric-TP/DCP on the same node, with GDN/Mamba state handoff.

**Fork status:** Implemented — single-node PD pair green (#99 M1/M2), token-vector KV
re-scatter, crash-robust.

**Engine comparison:**
- **SGLang:** yes (base) — sglang provides PD-disaggregation + the mooncake transfer stack; the
  single-node solo-prefill + asymmetric-TP/DCP decode + GDN-state handoff is the fork delta.
- **vLLM:** yes (base) — vLLM has P/D disaggregation; single-node heterogeneous solo-prefill
  split is not its target.
- **llama.cpp:** no — the RPC backend distributes whole layers/devices across hosts
  (pipeline-style), not a prefill-vs-decode role split; no PD-disaggregation concept found.
- **ik_llama.cpp:** no — same.

**References:** [SGLang PD disaggregation docs](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
· [Mooncake (FAST'25)](https://www.usenix.org/system/files/fast25-qin.pdf)

<a id="f15"></a>
### 15. Asymmetric-TP quantization correctness

**Feature:** asymmetric-TP quantization correctness + upstream quant bugfixes: GPTQ-MoE
`w2_scales` sharding fix at TP>1 (a genuine stock-sglang load defect, symmetric and asymmetric
TP); AWQ marlin MoE zero-point staging fix for `group_size < k-tile` (g=32); asymmetric-TP
`moe_wna16` K-mask illegal-memory-access fix; compressed-tensors and AutoRound-int4 asymmetric-TP
group alignment.

**Fork status:** Implemented — bugfixes #83, #85, GPTQ `w2_scales` (symmetric + asymmetric).

**Engine comparison:**
- **SGLang:** partial — AWQ/GPTQ/compressed-tensors/AutoRound exist upstream, but stock
  GPTQ-MoE fails to load at TP>1 (fork fixes it) and asymmetric-TP group alignment is absent.
- **vLLM:** partial — vLLM has its own Marlin/AWQ/GPTQ; the specific sglang GPTQ-MoE TP>1 defect
  is sglang-only.
- **llama.cpp:** n/a — asymmetric (head-based) TP is absent upstream there, so this bugfix class
  doesn't apply by construction.
- **ik_llama.cpp:** n/a — same; separately, its GPTQ/AWQ/Marlin-class kernels are an independently
  maintained lineage (not vLLM/SGLang's), so vLLM/SGLang-specific defects like the GPTQ
  `w2_scales` bug are inapplicable rather than avoided.

**References:** [SGLang quantization docs](https://docs.sglang.io/docs/advanced_features/quantization)
· [Marlin](https://github.com/IST-DASLab/marlin)

<a id="f16"></a>
### 16. Fast-lane priority scheduling

**Feature:** (`--enable-fast-lane`): an opt-in latency-priority class that preempts a tagged
request into the running batch, with a reserved-heavy-slots floor + heavy-aging to prevent
starvation; default off (default path unchanged).

**Fork status:** Implemented — Variant C Stage 0 (`--enable-fast-lane`, `--fast-lane-priority`,
`--fast-lane-reserved-heavy-slots`, `--fast-lane-heavy-aging-ms`).

**Engine comparison:**
- **SGLang:** partial — sglang has a priority scheduling subsystem the fork builds on; this
  reserved-floor fast-lane preemption class is the fork addition.
- **vLLM:** partial — vLLM has priority scheduling / preemption; not this reserved-floor
  fast-lane class.
- **llama.cpp:** no — `--prio`/`--prio-batch` set OS-level thread/process scheduling priority,
  not a request-level preemption/reserved-slots class; no evidence of queue-priority request
  scheduling in `tools/server`.
- **ik_llama.cpp:** no — checked directly in `examples/server/server.cpp`: the only "priority"
  mechanics found are an OpenAI-compat static JSON field (`{"priority": 0}`, unused for
  scheduling) and an internal cancel-task queue-front push used for the server's own bookkeeping.
  Neither is a user-facing request-priority preemption feature — confirmed not a false partial.

**References:** [SGLang server arguments](https://docs.sglang.io) · [llama.cpp `--prio`/`--prio-batch`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

<a id="f17"></a>
### 17. HiCache under asymmetric-TP/DCP

**Feature:** hierarchical KV cache: makes sglang's tiered KV cache (host-RAM L2 + file L3) correct
under non-uniform per-rank layouts — global->owned-compact KV index translation, a
storage-prefetch NCCL-deadlock fix under asymmetric-DCP, and a hybrid-SWA host-pool
over-allocation fix.

**Fork status:** Implemented — DCP index translation + prefetch-deadlock + host-pool fixes.

**Engine comparison:**
- **SGLang:** yes (base) — HiCache is upstream sglang; correctness under the fork's
  asymmetric-TP/DCP layouts is the fork delta.
- **vLLM:** n/a — vLLM uses LMCache / other KV-offload, not HiCache.
- **llama.cpp:** partial — `--prompt-cache` (disk file) + slot save/restore give an explicit,
  manually-triggered two-tier cache (RAM/VRAM + disk), not an automatic hierarchical hot/warm/cold
  tiering system like HiCache.
- **ik_llama.cpp:** partial — same explicit slot-save mechanism inherited, same limitation.

**References:** [SGLang HiCache docs](https://docs.sglang.io/docs/advanced_features/hicache)

<a id="f18"></a>
### 18. TP greater than num_kv_heads

**Feature:** replicated-KV + token-sharding: lets the tensor-parallel degree exceed the model's
KV-head count — and, via co-location, the physical GPU count (validated at TP=5, #62) — by
replicating the few KV heads and token-sharding the KV, including GQA re-grouping down to
single-head geometries.

**Fork status:** Implemented — validated TP=5 on 3 cards via co-location (#62).

**Engine comparison:**
- **SGLang:** partial — standard GQA already replicates KV heads when tp>num_kv_heads, but stock
  symmetric-TP still requires head-count divisibility; the fork combines replication with
  asymmetric-TP + token-sharded DCP + small (gqa=1) geometries.
- **vLLM:** partial — vLLM replicates KV under GQA similarly; not combined with asymmetric-TP /
  token-sharded DCP.
- **llama.cpp:** partial — `--split-mode row` parallelizes by matrix row rather than by attention
  head, which sidesteps the KV-head-count-divisibility wall entirely rather than solving it — a
  structurally different way of avoiding the same constraint. No token-sharded DCP or GQA
  re-grouping concept exists on top of it.
- **ik_llama.cpp:** partial, weaker confidence — same row-split mechanism inherited from
  llama.cpp, but per its own docs `-sm` there only lists `none`/`graph`/`layer`
  ([parameters.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md)) —
  **unverified** whether `row`-mode split (the part that matters for this row) is actually retained
  or was dropped in that fork's docs/code.

**References:** depends on features #1 / #2

<a id="f19"></a>
### 19. Broad model bring-up under asymmetric-TP

**Feature:** Qwen3.6-27B (GDN) and 35B-A3B (MoE) at asymmetric TP=3; Gemma-4 31B dense (triton
per-rank q-head workspace, EAGLE3 speculators-format #101) and 26B-A4B MoE SWA-hybrid
(vision-ignore mapper, gated-GeLU Marlin MoE, `--swa-pool-sizing` long-context cap); small /
small-head-count (2B, draft) models via replicated-KV GQA handling.

**Fork status:** Implemented — per-model; Gemma-4 EAGLE3 head fix (#101), 26B-A4B boot fix,
swa-pool-sizing.

**Engine comparison:**
- **SGLang:** n/a — model support; the asymmetric-TP variants + Gemma-4 speculators/SWA fixes are
  the fork's own code.
- **vLLM:** n/a — model support.
- **llama.cpp:** n/a — model support is model support; llama.cpp's own architecture coverage is
  very broad (arguably broader than SGLang/vLLM/this fork combined), but the row's fixes are
  specifically about asymmetric-TP interactions, which don't apply there.
- **ik_llama.cpp:** n/a — same; model coverage there is narrower than mainline llama.cpp but
  growing (DeepSeek, Qwen3/3.5, Gemma3/4, GLM-4/5, Kimi-2, Hunyuan per project README), and
  asymmetric-TP fixes are still n/a for the same reason.

**References:** [SGLang supported models](https://docs.sglang.io/supported_models/text_generation/generative_models.html)
· [Gemma docs](https://ai.google.dev/gemma)

<a id="f20"></a>
### 20. Session KV spill

**Feature:** (`--enable-kv-session-offload` + `--kv-session-offload-block-size` /
`-tick-interval` / `-restore-margin-tokens` / `-restore-hysteresis-steps`): the unit is a
**single in-flight sglang request** (one running sequence), **not** a multi-turn conversation —
note that "session" is separately used inside sglang itself
(`session/session_controller.py`, a parent-chained multi-request conversation), so the flag label
`session` is kept but here it means an in-flight request. On VRAM overflow (after tree eviction),
the **newest in-flight request's** full-attention KV shard is offloaded to host RAM while that
request **keeps decoding** via host-streamed attention; **FCFS-by-arrival** victim order (the
oldest running request stays fully resident) with fast-lane precedence; FIFO restore when device
capacity frees.

**Fork status:** Experimental — S1: single spilled request, eager path; overlap / multi-request
planned. **2026-07-25 additions, all green in the `integration/r3-probe` boot matrix, still
within the S1 (single-spilled-request) scope:** P2 host-RAM KV pool sized from a RAM budget
rather than a fixed count (arm B: 1.00 GB/rank of 24 GiB, unchanged accept level); prefill-side
spill, PS2 stage A+B' deep born-spilled extend (arm H: fired x3, `released device head=961`; a
born-spilled session's first tick is now deferred one iteration to fix an off-by-one where
`output_ids` is empty at that point); P1 configurable wave-back threshold (arm J: armed x3). A
separate defect found and fixed in the same pass: a spilled request could donate host "sentinel"
rows into the device radix tree as if they were real cache (`evictable=4351` against
`total=3600`), which also made the device-head-free logic a silent no-op — fixed by mirroring the
guard the finish-path already had (`c49472949a`). Two scenarios remain explicitly **not
validated**, carried forward as open: the spec-in-tick spill coincidence (never triggered in
testing) and 3-session co-residency (unreachable on this scheduler's admission formula,
demonstrated only at 2 co-resident sessions).

**Engine comparison:**
- **SGLang:** partial — on KV exhaustion sglang retracts a running decode request (frees its
  slots and re-prefills on re-admission; the CPU-copy path is not reloaded) and HiCache offloads
  only inactive prefixes; neither keeps a spilled session decoding.
- **vLLM:** partial — `--swap-space` + preemption (recompute / swap) free or swap out a request
  and resume/recompute it later; the request is paused, not decoded while its KV lives in host
  RAM.
- **llama.cpp:** partial — `-nkvo`/`--no-kv-offload` is a static, all-sessions-in-host-RAM setting
  decided at process launch, not a dynamic spill-on-overflow of just the newest request while it
  keeps decoding.
- **ik_llama.cpp:** partial — same `-nkvo`/`--no-kv-offload` flag inherited from llama.cpp, same
  static/global nature.

**References:** [vLLM preemption & swap docs](https://docs.vllm.ai/en/stable/configuration/optimization/)
· [SGLang retraction / conservativeness docs](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
· [llama.cpp `--no-kv-offload`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

<a id="f21"></a>
### 21. HTCCL cross-vendor collectives

**Feature:** vendor-neutral tensor-parallel collectives (env `SGLANG_HTCCL_TRANSPORT`, values
`gloo` / `shm` / `device`, transport registry): an all_reduce/all_gather/reduce_scatter/broadcast
implementation that never calls NCCL or RCCL, so one TP group can mix an NVIDIA and an AMD GPU —
something NCCL/RCCL structurally cannot do (no shared communicator across the two libraries). The
`device` transport does the reduction **on the GPU** over host-mapped memory
(`__threadfence_system` release/acquire, `clock64()`-deadline `__trap()` abort on a stuck peer
instead of a hang) and is CUDA-graph capturable; `gloo`/`shm` are host-staged fallbacks that are
rejected at startup under CUDA graphs (`9adce4b551`) rather than crashing mid-capture.

**Fork status:** **Implemented and validated — status upgraded 2026-07-25** from the prior
"planned/prototyped, not landed." Merged into `integration/r3-probe` (`73679d6b47`,
`9a10846a82`, plus the `feat/htccl-gfx900` follow-on `aec1308973`, 7 of that branch's 9 commits).
Correctness: known-answer tests per collective / dtype (fp32, bf16, fp16) / world-size / transport
against `torch.distributed`, plus a standalone 3-rank ground-truth probe on all three transports.
Two real bugs were found and fixed pre-cross-vendor: an output-buffer aliasing defect where two
same-shape `all_reduce` results silently became the *same tensor* on `gloo`/`shm` (found
independently on a second host running an older branch, traced to a predecessor commit that
predated the fix, not a new instance), and `reduce_scatter` scattering the wrong axis for
`dim >= 2` (`movedim(0, dim)` moves axis 0 *to* `dim`, which only coincides with "bring axis `dim`
to front" for `dim` in {0, 1} — silent above that, fixed in both transports, `8acd4221a3`). A
CUDA-graph-capture-time broadcast (the speculative draft-pick sync) was made capturable via
`all_gather` + a source slice, byte-exact for the int64 index it carries (`e9678aa798`).
**Cross-vendor (RTX 2080 Ti sm75 + Radeon RX Vega 64 gfx900), eager:** collective-level byte-exact
against `torch.distributed`; model-scale byte-identical to `gloo` on the same pair (Qwen2.5-0.5B,
then Qwen3.5-4B even 2/2 **and** uneven 3,1 TP splits, token ids and chat text, 2026-07-25);
throughput at model scale (Qwen3.5-4B, decode by slope over 32/288 new tokens, prefill from a
1172-token prompt): `device` transport **+37% (even) / +48% (uneven) decode, +45%/+62% prefill vs
`gloo`** on this pair (16.51 vs 11.13 tok/s uneven-decode) — the largest single lever measured on
that host. **Cross-vendor with CUDA graphs remains "in reach, not demonstrated"**: torch-level
graph capture works on both vendors individually, but sglang's own graph runner hit three more
CUDA-only assumptions in sequence on gfx900 before a fourth, real root cause was isolated and
fixed (`cudaLaunchConfig_t`/`cudaLaunchAttribute` types absent from ROCm 6.3, `f560631dc6`; the
CUDA-graph phase plan silently diverging per-rank instead of being a group decision, `55bfdb4db8`;
ROCm's launch-error-check strategy performing an extra post-launch query illegal during capture,
`2548b630bf`; and finally an always-on device-side `assert()` in the KV-cache store kernel that
makes the kernel **fail to launch at all** on gfx900, independent of streams or capture, fixed
with `SGL_DEVICE_ASSERT`, a guarded `__builtin_trap()` on ROCm, byte-verified both directions,
`fa5c507476`). That last fix is on `feat/htccl-gfx900` (tip `3cc2fc9da5`) and is **not yet
merged** into `integration/r3-probe`; symmetric decode capture on both ranks, and the still
separately-registered NVIDIA-side prefill-capture assertion, remain open after it.

**Engine comparison:**
- **SGLang:** no — no cross-vendor NVIDIA/AMD TP path; SGLang's distributed backends are NCCL
  (NVIDIA) / RCCL (AMD), never bridged.
- **vLLM:** no — same; vLLM's distributed layer is likewise NCCL/RCCL, not bridged across
  vendors.
- **llama.cpp:** partial — the RPC backend (`tools/rpc`, `ggml-rpc-server`) lets one
  `llama-cli`/`llama-server` process delegate compute to remote/heterogeneous backends (CUDA,
  Metal, CPU, and presumably Vulkan/ROCm devices too) over TCP, without needing NCCL/RCCL. But
  it's explicitly "proof-of-concept... fragile and insecure" per its own README, and it's a
  backend-delegation/pipeline model (the local process dispatches ops to a remote device), not a
  TP all-reduce/all-gather collective substituting for NCCL *within* one TP group — a materially
  different mechanism from HTCCL's collective-replacement approach.
- **ik_llama.cpp:** partial — RPC backend confirmed present and maintained
  (`ggml/src/ggml-rpc.cpp`, project history includes "RPC sync" #193 and "RPC improvement" #480),
  same mechanism and same POC-grade caveat as llama.cpp.

**References:** [NCCL](https://docs.nvidia.com/deeplearning/nccl) · [RCCL](https://github.com/ROCm/rccl) · [Gloo](https://github.com/pytorch/gloo)
· [PyTorch distributed backends](https://pytorch.org/docs/stable/distributed.html) · [UCX](https://openucx.org)
· [llama.cpp RPC backend README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)

<a id="f22"></a>
### 22. fp8 dequant fallback (W8A16)

**Feature:** fp8 checkpoints on GPUs without a native fp8 GEMM, via a dequant W8A16 path
(compressed-tensors `CompressedTensorsW8A16Fp8`): a third fp8-serving tier alongside native W8A8
(sm89+/MI300+) and the existing Marlin W8A16 fallback (CUDA-only, sm80+) — weights stay fp8 in
VRAM and are dequantized to the compute dtype per forward; activations are never quantized. Gated
by a **functional** capability probe (`fp8_native_gemm_available()`, a cached trial
`torch._scaled_mm`) rather than a capability-number comparison, because
`torch.cuda.get_device_capability()` reports `(9, 0)` for **both** Hopper and AMD gfx900 — a
comparison-based gate let an 8 GB Vega 64 pass a `>= 89` fp8 threshold it cannot execute (would
have died later on the missing kernel instead of being refused at startup; the 4th instance of
this same `(9,0)` collision found on this branch).

**Fork status:** **Implemented, GPU-validated cross-vendor — 2026-07-25, on `feat/htccl-gfx900`
(`3cc2fc9da5`, tip of that branch), NOT YET merged into `integration/r3-probe`.** Verified CUDA
is untouched by construction (ROCm branch is an early return) across sm120/sm86/sm75/gfx900.
Correctness: Qwen3.5-4B-FP8-dynamic, same 4 prompts greedy — solo Vega 64 (gfx900) vs solo RTX
2080 Ti (sm75) **byte-identical**; solo Vega 64 vs mixed TP=2 uneven `3,1` **byte-identical**;
solo Vega 64 vs mixed TP=2 even 2/2 **byte-identical**. Capacity claim measured, not promised: the
model fits solo on the Vega 64 in fp8 (weights 6.27 GB, 1.07 GB free after KV) where fp16 does not
fit at all (8.8 GB against 8.0 GB VRAM). **Honest counter-reading:** the dequant fallback costs
**~23% of decode** vs fp16 at the same TP config (12.67 vs 16.51 tok/s), next to nothing on
prefill (966 vs 982 tok/s); and on *this specific pair*, fp8 makes the mixed pair pointless — the
model now fits solo on the 2080 Ti alone at 15.23 tok/s, which beats the mixed pair's 12.67, so
the pair is only useful when the model does not fit on either card alone (the intended case, e.g.
27B/35B, untested here). **Explicitly OPEN:** the `fp8.py` (non-compressed-tensors) `Fp8Config`
family — the family the user's own 27B/35B checkpoints use — is **not wired to the new probe**;
per-tensor fp8 checkpoints and `CompressedTensorsW8A8Fp8MoE` (fp8 MoE) are untested on this path.

**Engine comparison:**
- **SGLang:** no — no dequant fallback for fp8 checkpoints on sub-sm80/non-MI300 hardware; such
  hardware cannot serve an fp8 checkpoint at all.
- **vLLM:** no — vLLM's fp8 paths similarly require a native GEMM or Marlin (sm80+); no plain-torch
  dequant fallback for older/cross-vendor cards.
- **llama.cpp:** partial — handles fp8 checkpoints via **offline conversion**, not a runtime
  dequant-fallback GEMM path: `convert_hf_to_gguf.py --fp8-as-q8` dequantizes FP8 weights at
  conversion time to Q8_0 (or BF16/F16 by default), producing an ordinary GGUF that the existing
  kernels serve unmodified. There is no functional capability probe and no per-forward dequant —
  the fp8 handling happens once, at conversion time, entirely outside the serving engine. Confirmed
  directly in the local checkout (`convert_hf_to_gguf.py` line 152-153, `--fp8-as-q8`).
- **ik_llama.cpp:** unverified — no equivalent conversion-time fp8 handling was confirmed in the
  time available; ik_llama.cpp typically consumes GGUFs already produced by mainline llama.cpp's
  converter, so it may simply inherit this rather than needing its own path, but that was not
  independently checked.

**References:** [compressed-tensors fp8 overview](https://docs.vllm.ai/en/latest/features/quantization/int8.html)
· [Marlin](https://github.com/IST-DASLab/marlin) · [torch `_scaled_mm`](https://pytorch.org/docs/stable/generated/torch.Tensor._scaled_mm.html)
· [llama.cpp `convert_hf_to_gguf.py --fp8-as-q8`](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)

<a id="f23"></a>
### 23. Turing/gfx900 without sgl-kernel

**Feature:** lets the server start and serve on GPUs `sgl-kernel`'s prebuilt wheel does not cover
at all (sgl-kernel is cubin-only, floor sm80 — an RTX 2080/2080 Ti/T4 has literally no executable
code in the package) by gating every affected compute path on a **two-level capability
predicate** — `sgl_kernel_importable()` (import time, must not touch the device) and
`sgl_kernel_runnable()` (first use, catches installed-but-wrong-arch) — instead of `is_hip()` /
platform. That platform-vs-availability confusion was wrong in **both** directions on this
branch: true by accident on gfx900 (whose own ROCm build only covers gfx942/gfx950) and false by
accident on sm75, for the identical underlying problem on two different vendors. Real fallbacks
are provided, not just guarded imports: `RMSNorm.forward_cuda` -> `forward_native` (mirroring the
existing HIP-fallback route), the sampler's `flashinfer` backend degrading to the existing
torch-native `pytorch` backend, `rope`/`clamp_position` routed by kernel availability rather than
platform.

**Fork status:** **Implemented and GPU-validated on both vendors, 2026-07-25** (`0eb7e68880`
Turing support; `3f0a93ac1c` rope/clamp_position availability routing; `621311aa24` a fourth
instance of the same platform-vs-availability bug, this time hitting the **NVIDIA** rank —
`GemmaRMSNorm.forward_cuda` lacked the sm80 guard its sibling `RMSNorm` already had). Verified
end-to-end on a real RTX 2080 Ti with `sgl_kernel` absent: all 11 core modules import, the server
starts, greedy generation is coherent; 608 unit tests pass. Numerically: the native fallback is
not byte-identical to the kernel path (different reduction order, ~4.8e-07-class difference), but
`forward_native` was measured **byte-identical between sm75 and gfx900** on identical inputs.
Solo Vega 64 (gfx900): coherent output on both `torch_native` and sglang's Triton backend,
matching token ids to a solo sm75 run; mixed-vendor TP=2 (both ranks on Triton, HTCCL/gloo, 8192
ctx) reproduced the same token ids as both solo runs. **Scope note:** Triton itself compiling for
gfx900 depends on the third-party `Said-Akbar/triton-gcn5` fork (stock Triton refuses gfx900
outright) — an external prerequisite, not fork code.

**Engine comparison:**
- **SGLang:** no — no capability-fallback path; `sgl-kernel`'s absence or wrong-arch state is
  either not detected or fatal upstream.
- **vLLM:** no — vLLM has its own per-kernel capability gating, but not this specific two-level
  import/runtime split for `sgl-kernel`-class dependencies; not applicable in the same form.
- **llama.cpp:** partial (same outcome, different reason) — llama.cpp's CUDA/HIP/Vulkan kernels
  are compiled from source for a broad architecture range from the start; there is no separate
  cubin-only "sgl-kernel"-class package with a floor above these GPUs to begin with. The Vulkan
  backend additionally runs on almost any GPU (AMD/Intel/old NVIDIA). So the base-operation
  *outcome* (serving on Turing/gfx900) is native and has been all along, but the row's specific
  two-level capability-predicate machinery doesn't exist because there is no gated package to
  detect around — a "we never had the problem" partial, not a "we solved the same problem" yes.
- **ik_llama.cpp:** partial — same reasoning as llama.cpp; additionally its CPU-first design
  (row-interleaved quant kernels tuned for CPU/hybrid inference) makes old/low-VRAM-GPU + CPU
  hybrid operation an explicit design target rather than a fallback path.

**References:** [NVIDIA Turing architecture](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth)
· [AMD gfx900 (Vega)](https://gpuopen.com) · [Triton](https://github.com/triton-lang/triton)
· [triton-gcn5 (external, gfx900 support)](https://github.com/Said-Akbar/triton-gcn5)

---

## Guarded / descoped (implemented in code, deliberately gated off)

These were built and evaluated in code, then gated off — listed for completeness because real
code work went into them, but they are not shipped as usable capabilities. (No llama.cpp /
ik_llama.cpp comparison is added here: these are internal fork decisions about the fork's own
uneven-TP/DCP machinery, which has no upstream analog in either engine — see rows 1/2/18 above
for that comparison.)

- **Tree speculative decoding with `--speculative-eagle-topk > 1` under asymmetric-weighted DCP
  (#76)** — implemented and GPU-tested; found silently non-greedy under weighted DCP and
  perf-negative on this PCIe rig; restored as a hard fail-fast guard with a CPU test.
- **SWA-DCP Stage B** — **not implemented.** (Corrected 2026-07-25: this row previously read
  "implemented and evaluated (~+6-10%)". It was never built; the DCP Triton extend path still
  raises `NotImplementedError` for sliding windows, and the ~+6-10% figure is an *ex-ante estimate
  from the design doc*, not a measurement.) Descoped as not worth the cost; Gemma-4 SWA
  long-context is served instead by the `--swa-pool-sizing` cap (row 19).
- **Replicated-KV eligibility widened to `kv == tp` (the `<` -> `<=` flip, row 1)** —
  2026-07-25: implemented, red/green-tested on CPU, and validated on GPU; the GPU measurement
  **refuted** it. At `kv == tp`, groups == ranks always, so the kv-group alignment repair that
  makes uneven splits work at `kv < tp` has no room to operate, and the raw (unaligned) split
  straddles a kv-head boundary for any non-uniform ratio — boots, then dies on the first forward
  (`REPLICATED-KV current-chunk attention (#105)` error). The existing `<` semantics were kept,
  now with the measured rationale pinned in a test
  (`test_kv_eq_tp_stays_in_normal_mode_by_measurement`); genuinely uneven attention at `kv == tp`
  would need a ragged kernel supporting per-rank non-uniform GQA mapping, not a threshold change.

## Cross-vendor bring-up (2026-07-25): additional bugfixes with upstream relevance, and non-defects

Found and fixed during the HTCCL / cross-vendor campaign (rows 21-23 above), listed here because
each is a genuine, independently-triggerable defect class rather than a one-off — several recur
in more than one component, which is the point of recording them together. (These are SGLang/
Triton-backend-internal defects; llama.cpp and ik_llama.cpp use an entirely different compute
stack (ggml), so no comparison column applies to this section.)

- **Even-DCP under the Triton backend silently corrupts output when KV heads are not replicated
  across the DCP group** (`6a8e7f76ef`) — root-caused via a discriminator matrix on real models:
  the decode path all-gathers the whole DCP group's q heads and remaps them onto *this rank's*
  local kv-head shard, which is only correct when `tp_size // total_kv_heads >= dcp_size` (the
  geometry of the path's origin, upstream `sglang`
  [#25090](https://github.com/sgl-project/sglang/issues/25090), Qwen3.5-397B TP=8/DCP=2 — correct
  only for that specific CI geometry, not in general). Now rejected at backend construction with
  the numbers and three ways out; the fork's own uneven-DCP geometry is exempted (different,
  aware, head-handling). Left explicitly open: the uneven-DCP + dense-model-class combination
  remains unguarded (measured to also produce mojibake), and flashinfer silently no-ops plain
  even `--dcp-size N` instead of erroring — both registered as separate open items, not fixed
  here.
- **`o_proj` reject-guard for uneven-TP-unaware attention classes** (`dd68fad951`, folded into
  row 1) — clamping `o_proj`'s input to the head-aligned split would have traded a loud shape
  error for a silent wrong-numerics one on 3 model classes; rejected at construction instead.
- **The `--rank-kv-ratio` / uneven-DCP arg gate silently ignoring an unusable token vector, now a
  hard reject** (`4c90038a78`, folded into row 2).
- **`GraphSharedOutput` (the shared decode/prefill/eagle-draft-extend logits buffer) —
  investigated as a suspected third instance of the "shared buffer handed to a live caller" defect
  family, and confirmed NOT a defect.** A falsifier was built first (an unshared, faithfully-sized
  variant reproduces the shared run bit-for-bit; only an under-allocated naive variant diverges,
  which identifies the allocation size as the actual variable, not sharing). Structurally: the
  buffer is obtained once per runner in `__init__`, not per call, so there is no second call that
  could hand it to a live consumer — the standard CUDA-graph static-buffer pattern. Verified for
  the overlap scheduler ON (the shipped configuration); `return_logprob`'s separate read path was
  not exercised and is stated as a limit of the check, not swept under it.
- **Validator hygiene**: the mechanical output-corruption validator built for this campaign
  scored its own first test case wrong — a healthy, math-heavy repetition sample as `CORRUPT`,
  because its letter-fraction rule fired on digit/punctuation-heavy tokens. The rule was
  **removed rather than tuned** (word-level metrics govern when they are informative; repetition
  is reported separately from corruption rather than folded into one score) — a validator that
  manufactures the error it exists to catch is worse than no validator.

## Scope note

This matrix lists only capabilities with landed code (verified directly against
`integration/r3-probe` and, where noted, the not-yet-merged `feat/htccl-gfx900`). Internal-roadmap
items that are planned or only partially prototyped — e.g. a host-RAM tiered-KV fabric for the
weightless lane, downloading VRAM, a draft-KV-pool DCP layout, symmetric cross-vendor CUDA-graph
capture (row 21), and the `fp8.py` `Fp8Config` family on non-CUDA-native hardware (row 22) — are
intentionally excluded until they land.

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

**Heterogeneous parallelism and context parallelism**
[HexGen](https://arxiv.org/abs/2311.11514) · [Hetis](https://dl.acm.org/doi/10.1145/3712285.3759784) · [Tangram](https://arxiv.org/pdf/2606.16907)
· [Cronus](https://arxiv.org/pdf/2509.17357) · [Helix](https://arxiv.org/pdf/2507.07120) · [Medha](https://arxiv.org/pdf/2409.17264)
· [Context Parallelism for Million-Token Inference](https://arxiv.org/pdf/2411.01783)
· [vLLM parallelism scaling docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling) · [SGLang DCP issue](https://github.com/sgl-project/sglang/issues/12196)
· [vLLM context parallel deployment](https://docs.vllm.ai/en/main/serving/context_parallel_deployment.html)

**MoE expert offloading**
[KTransformers](https://github.com/kvcache-ai/ktransformers) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
· [MoE-Infinity](https://arxiv.org/html/2401.14361) · [SGLang offload PR](https://github.com/sgl-project/sglang/pull/3675)
· [SGLang expert-granular request](https://github.com/sgl-project/sglang/issues/14233) · [vLLM offload config](https://docs.vllm.ai/en/latest/api/vllm/config/offload/)

**KV/attention disaggregation**
[Adrenaline](https://arxiv.org/pdf/2503.20552) · [Mooncake](https://www.usenix.org/system/files/fast25-qin.pdf) · [CrossPool](https://arxiv.org/pdf/2606.24506)

**Checkpoint/restore and memory**
[ServerlessLLM](https://arxiv.org/pdf/2401.14351) · [Tangram](https://arxiv.org/pdf/2512.01357) · [PipeBoost](https://arxiv.org/pdf/2503.17707)
· [vLLM CUDA checkpoint/restore RFC](https://github.com/vllm-project/vllm/issues/34303) · [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
· [vLLM CuMemAllocator](https://docs.vllm.ai/en/latest/api/vllm/device_allocator/cumem/) · [SGLang offload/wake-up PR](https://github.com/sgl-project/sglang/pull/19152)
· [vLLM optimization config](https://docs.vllm.ai/en/stable/configuration/optimization/)

**GGUF**
[vLLM GGUF docs](https://docs.vllm.ai/en/stable/features/quantization/gguf/) · [llama.cpp](https://github.com/ggml-org/llama.cpp)
· [GGUF format spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) · [Qwen3.5 GGUF evals](https://kaitchup.substack.com/p/more-qwen35-gguf-evals-and-speculative)

**Determinism**
[SGLang deterministic inference docs](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/deterministic_inference.md)
· [SGLang determinism issue](https://github.com/sgl-project/sglang/issues/10278) · [SGLang determinism blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic)
· [Batch Speculative Decoding Done Right](https://arxiv.org/pdf/2510.22876) · [Bit-Exact AI Inference Verification](https://arxiv.org/pdf/2606.00279)

**Device identity and telemetry**
[NVIDIA MPS](https://docs.nvidia.com/deploy/mps) · [NVML API](https://docs.nvidia.com/deploy/nvml-api) · [NVIDIA DCGM](https://developer.nvidia.com/dcgm)
· [vLLM metrics design](https://docs.vllm.ai/en/latest/design/metrics/)

**Cross-vendor collectives and low-end/AMD GPU support**
[NCCL](https://docs.nvidia.com/deeplearning/nccl) · [RCCL](https://github.com/ROCm/rccl) · [Gloo](https://github.com/pytorch/gloo)
· [PyTorch distributed backends](https://pytorch.org/docs/stable/distributed.html) · [UCX](https://openucx.org)
· [NVIDIA Turing architecture](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth) · [AMD gfx900 (Vega)](https://gpuopen.com)
· [Triton](https://github.com/triton-lang/triton) · [triton-gcn5 (external)](https://github.com/Said-Akbar/triton-gcn5)
· [Marlin](https://github.com/IST-DASLab/marlin) · [torch `_scaled_mm`](https://pytorch.org/docs/stable/generated/torch.Tensor._scaled_mm.html)
· [llama.cpp RPC backend](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)

---

## Changelog

**2026-07-25 update, pass 2** (this pass): added two comparison engines, `llama.cpp` and
`ik_llama.cpp`, across every row and (where sensible) noted in the guarded/descoped and
cross-vendor-bugfix sections why no comparison applies there. Sources: a local `llama.cpp`
checkout (`/spinning/llm_stuff/llama.cpp-master`, commit `0c4fa7a989`, 2026-07-12) read directly
for CLI flags, model architecture files, and conversion-script behavior; GitHub API/raw-file
fetches against `ikawrakow/ik_llama.cpp` `main` (2026-07-25) for the same, since no local
checkout exists; WebSearch/WebFetch for project docs, DeepWiki, and discussion threads where
neither repo answered directly. Distribution across the 28 rows: llama.cpp 4 yes / 11 partial /
8 no / 5 n/a; ik_llama.cpp 3 yes / 11 partial / 8 no / 5 n/a / 1 unverified. Several `partial`
verdicts required explicitly naming a *different mechanism* per the coordinator's brief (e.g. row
1: llama.cpp's `--tensor-split` is a row/layer split, not head-based TP; row 21: llama.cpp's RPC
backend is backend-delegation/pipeline, not a collective substituting for NCCL; row 23:
llama.cpp/ik_llama.cpp never had the cubin-only-package problem to begin with, so their "partial"
reflects a mechanism gap even though the end-user outcome already matches). Two items are marked
`unverified` rather than guessed: whether ik_llama.cpp's `-sm` still exposes `row`/`tensor` split
modes (rows 1, 18), and whether ik_llama.cpp has any fp8-checkpoint conversion path of its own
(row 22).

Separately, this pass **restructured the whole file** for GitHub-width readability: the previous
single wide table (SGLang/vLLM columns only, some rows exceeding 3000 characters in one line) is
replaced by a compact overview matrix (verdict tokens only, linked by anchor) plus one `###`
detail section per feature carrying the full description, fork status, per-engine breakdown, and
references. No content was removed in this reflow — every measurement, caveat, and reference from
the prior version is present in a detail section, including the honest-status phrasings the
2026-07-25 pass-1 changelog entry (below) already called out. Reference links were also converted
from bare `domain/path` mentions to titled Markdown links per the same readability goal.

**2026-07-25 update, pass 1**, verified against `integration/r3-probe` (`wt-merge-probe`, local
HEAD `aec1308973`, pushed to at least `4c90038a78`) and `feat/htccl-gfx900` (`wt-htccl`, tip
`3cc2fc9da5`) — not against `/spinning/htsglang`, which is a stale checkout, nor against memory
files or task lists except as a secondary cross-check:

- **Added**: row 21 HTCCL (vendor-neutral TP collectives — status upgraded from "planned,
  excluded" to "implemented and validated," including the still-open cross-vendor-CUDA-graph
  chain); row 22 fp8 W8A16 dequant fallback for sm75/gfx900 (implemented, GPU-validated
  cross-vendor, **not yet merged** into `integration/r3-probe`); row 23 Turing (sm75) / gfx900
  base operation without `sgl-kernel`; a "Cross-vendor bring-up: additional bugfixes with
  upstream relevance, and non-defects" section (even-DCP/Triton kv-replication guard,
  `GraphSharedOutput` cleared as a non-defect, validator hygiene fix); a "guarded/descoped" entry
  for the `kv == tp` replicated-KV flip (implemented, GPU-measured, reverted).
- **Corrected / updated in place**: row 1 (o_proj-vs-head-split audit and reject-guard; DFLASH
  per-rank uneven-TP shards); row 2 (DCP arg-gate audit confirming it is a real wiring
  dependency, not arbitrary; silent-ignore -> hard-reject hardening; even-DCP/Triton
  kv-head-replication defect cross-referenced); row 3 (`--rank-tp-ratio`/`--rank-kv-ratio`
  decoupled from `--rank-gpu-id`, enabling the cross-vendor placement model); row 5 (lazy
  single-graph capture + DFLASH context-retirement, #156-4, merged and validated); row 8d (GGUF
  MMQ decode threshold #163: exact measured gain +9.7-10.6% on the one rank it affects, and the
  fact that flag-ON output is **not** byte-identical to flag-OFF — stated because the earlier
  draft of this update nearly reported it as a clean speedup without that caveat); row 20
  (KV-session-offload S1 scope: P1/P2/PS2 additions, all green, still S1-only; a sentinel-row
  radix-tree leak found and fixed).
- **Removed / walked back**: the Scope note's listing of "cross-vendor HTCCL AMD bring-up" as an
  excluded, not-yet-landed item — it has landed and is now row 21; replaced with the items that
  are genuinely still open (symmetric cross-vendor CUDA-graph capture, the `fp8.py` `Fp8Config`
  family).
- **Verified unchanged, not touched**: the 2026-07-25 SWA-DCP correction already present in this
  file (row "Guarded / descoped") was re-checked against the current branches and still holds as
  written; not re-stated as new.
