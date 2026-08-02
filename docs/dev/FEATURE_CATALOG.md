# Feature Catalog — what this fork already has

Read this BEFORE searching the tree or building anything: most capabilities you
are about to look for already exist. Rules: (1) never declare something
"impossible" or "missing" without checking this file, FEATURES_VS_UPSTREAM.md
and `git log`; (2) whoever merges a new feature updates the matching section in
the SAME merge. Last full refresh: 2026-08-02 (tip 33148dbe0f).

## 1. Uneven parallelism (core differentiator)
- **Uneven TP** `--rank-tp-ratio` + `--rank-gpu-id`: per-card weight shards.
  `auto` = byte-proportional from NVML totals minus auto reserve; with
  `--rank-perf-tune both|dec|enc|maxkv` the planner solves the vector.
  Unit system: `tp_units`/`tp_family` per layer class (16-element MLP family;
  coupled-dim rule: gate_up output and down_proj input partition the SAME
  intermediate dim and must coarsen identically); per-layer family table for
  `block_configs` models (Nemotron-Puzzle class).
- Sibling flags: `--rank-mlp-ratio`, `--rank-vocab-ratio`, `--rank-moe-ratio`
  (experts BETWEEN ranks), `--rank-moe-resident-fraction` (GPU/host split
  WITHIN a rank), `--rank-kv-ratio` (`coupled|speed|vector` — decouples KV
  split from weight split), `--rank-auto-reserve-mib`, `--rank-gpu-memory-mib`
  (absolute per-rank MiB budget with a line-item ledger incl. lane pools).
- **Uneven DCP** (`dcp_size` + token vector): token/KV sharding across ranks,
  weighted owner rule, SWA-hybrid support, TP>kv_heads via replication+token
  shard. **Draft-KV-DCP**: draft KV token-sharded (−67 % draft KV; above
  TP>kv_heads, replicated is the DEGRADED layout). LSE log base follows the
  attention backend (FlashMLA = natural log).
- **TPxPPxTP**: pipeline across rigs with per-stage TP groups. Slices 1+2
  merged (cross-rig PP=2 over 40G, full decode graphs on both stages incl.
  sm75). Slice 3 built (branch `feat/tpxppxtp-slice3-201`): world-MIN
  `max_total_num_tokens` before the reduce, `--pp-stage-ratio`
  (score-proportional, snaps to full-attention boundaries), stage-local mamba
  slots, `auto` under PP with an agreement gate, `SGLANG_PP_SHAPE_CACHE`.
- **TP5+ emulation** via NCCL multi-rank co-location (several ranks per card).

## 2. Planner / solver
Key solver: water-filling over an affine cost model, pair-matrix collective
term, roles/nesting as box bounds, Pareto+knee, admissibility gates,
`coresident_budgets()`. Measured phase optima: prefill 10,1,1 (+ decoupled KV
2,11,10 keeps capacity), decode ~3,2,2. `--objective energy` end to end with
refusal over silent substitution. `planner/rejected.py` = machine-readable
register of discarded approaches — check it before re-proposing anything.

## 3. Memory tiers / offload / spill
- **Expert offload**: MoE experts in a pinned host-RAM pool, streamed over
  PCIe on demand. **CUDA-graph-compatible path EXISTS** (decode-graph +
  eager-prefill hybrid): GPU kernels read the pinned pool via UVA zero-copy
  indexed by device-resident router ids — no host sync. The DeepSeek-V4 GGUF
  path still uses a `tolist()`-syncing variant: that is a PORTING item, not a
  wall. Double-buffered prefetch with compute overlap; expert-major prefill
  waves (`SGLANG_MOE_OFFLOAD_WAVE_ORDER`, byte-identical proven); fp8 presplit;
  load-time-aware halves for fp8/GPTQ/AWQ (GGUF-MoE half missing — guarded);
  `SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` = frozen-resident-set escape hatch (refusal
  until its byte gate is green).
- **#394 cold-shard chain** (slice 1 merged): measured H2D provenance chain
  (env > card-probe > nvml-negotiated > refusal; `absent` unselectable),
  `cold_tier_shm.py` shared-DRAM segments (UUID/BDF identity, manifest read
  lazily after load, header sealed last, PROT_READ views with kernel-enforced
  write protection), boot-time refusal for delegation on disjoint expert
  shards. Fetch-path wiring (slice 2) open. Graphs incl. CPU-MoE are
  IMPLEMENTATION EFFORT, not blocked: UVA reads, cudaGraph host nodes,
  CUDA>=12.4 conditional nodes, and graphs pin ADDRESSES not CONTENTS
  (spill/restore under fixed buffers is legal).
- **HiCache** L1-L3 prefix cache (validated with uneven DCP/TP; storage key
  includes kv-dtype; runtime attach/detach works on UnifiedRadixCache).
- **KV session offload (kvso)**: FCFS spill of youngest sessions to RAM (KV
  only, GDN stays resident), budgets (volume/rate/window, demote to HiCache),
  idle-first victim choice, decoupled from speculation.
- **Hibernate to disk** (weights+KV survive process exit; uneven-TP3 reload
  50s→8-14s) + suspend-to-RAM (memory saver).
- **Runtime VRAM dial** per card (VMM page return), **KV pressure ladder**
  (geometry stages instead of rejects; explicit ladders work, but
  `--kv-pressure-ladder auto` is currently BROKEN — hard-fails at runtime,
  audit #421 F1; rung-dependency refusals exist and fire), **KV resharding**
  at phase boundaries (delta move <1 s, `kv_reshard_vectors`), **GDN slot
  ladder** (resident-state cap + idle vacate → VRAM back to KV pool).
  WARNING (audit #421 F2): `--lane-offload-profile/-class-policy/-park-targets`
  are advertised in CLI help but currently DISCARDED (register never
  configured from ServerArgs) — do not rely on them until wired.
- **memtier registry**: tier ids with volatility + payload class and
  provenance `measured|estimate|absent` (absent refuses use). HONEST STATE
  (audit #421): ZERO consumers wired today — "all consumers pick targets from
  it" is the TARGET rule, not the current state; existing offload/spill paths
  still carry their own target lists.

## 4. Speculative decoding
NEXTN/MTP standard (steps 3, topk 1, draft 4); adaptive draft length (upstream
base, fork adds graph-offload, high-accept ladder, frozen-MTP, hetero
determinism); acceptance-driven DFLASH<->NEXTN switch + adaptive k; DFLASH solo
draft on the big card (vocab broadcast reclaims ~5 GB); chain-spec on the
weightless lane; multi-layer EAGLE fixes; spec-algo name validation (one
source, parse-time refusal); canonical `--speculative-draft-model-path`.
Tree-spec topk>1 under DCP is HARD-GATED (silently wrong + perf-negative — do
not re-attempt without new evidence; see rejected register).

## 5. Multi-group runtime (dual lane)
Slices A-D merged: lane-correct context overlays (~370 callsites), own thread +
high-priority stream, lend/reclaim in ms, SM-contention pairing rule,
lane-NEXTN head. Lane spec chain merged: rank-local draft-KV sizing, chain-spec
topk=1 on the lane, lane prefill chunking (`dual_group_lane_prefill_chunk`;
spec chunks carry the head primer — costs measured), Marlin LoRA workspace
keyed (lane,name). PD disaggregation: prefill satellite carries hybrid GDN
(KV+mamba slot via mooncake), default graph-covered.

## 6. Weightless KV lane
A card holds ONLY KV + attention (no weights): chunked prefill/extend, fp8/int4
worker KV, DCP comm fusion, graph-captured streaming decode, host-tier KV
spill, chain spec. **Live session handover without server stop** + draft
re-sharder as its own spec type: BRANCH-ONLY, NOT on the integration tip yet
(branch `feat/live-handover-261`, `POST /session_handover`, five-phase at
session scope, hard GDN-blob gate; merge pending its GPU byte gate).

Also wired on the tip but easy to miss (audit #421): the regime-controller
gate machinery, KV-pressure rung-dependency refusals, the hibernate flag
contract (`hibernate_dir` + weights/draft CPU/disk backup flags), and a
118-name retired-env guard that refuses stale SGLANG_* variables loudly.

## 7. Collectives / transport
**barlink** (own vendor-neutral CCL): NCCL-parity device transport,
cross-vendor byte-exact, UCX transport (chunk pipelining, dual worker), tuned
all_gather ring, graph-capable direct mode. **Smallbar BAR1 direct path**:
peer VRAM over 256-MiB BARs, beats NCCL 1.13-1.34x in serving.
`--collective-net-small/-bulk` per message class with typo hard-reject.
dmabuf GPU-RDMA works on consumer cards with the stock driver. Rig facts: NO
P2P/NVLink here, negotiated PCIe x4/x8/x8 (NVML max-width reports x16
NAMEPLATE — always read negotiated width), NCCL-verbs broken on our RoCE.

## 8. GGUF stack
Generalized loader (registry + family mapping tables), unsloth-UD, mixed-dtype
fused GDN qkvz, MoE tensor mapping, vision/mmproj, sibling-config validation,
DeepSeek-V2/3/4 class GGUF-safe (`.qweight` accessors, quantization_config
drop, tokenizer route). Perf: batched MMVQ, Q8 lm_head, K-quant MMVQ tuned to
Q8_0 efficiency (TP=2 beats llama.cpp), graph-replay numeric safety for ALL
quants, `gguf_mmq_decode_threshold`.

## 9. Quant lanes
FP8 (sm120 GEMM tuned; per-channel fused GEMV; opt-in deterministic
`SGLANG_DETERMINISTIC_FP8_GEMM`; e4m3 KV bit-exact on sm86), INT8-W8A8 (default
recommendation; sm86-native lane; beware the dual-dist wheel trap — pin by
sha256), NVFP4 (V4 class usable via dequant fallback for unpackable layers),
Marlin alignment family (SEVEN sibling bugs fixed — device-free fold predicate,
lcm=128 on coupled dims; alignment fixes must preserve cross-layer agreement).

## 10. Determinism / quality gates
Hetero-determinism roots fixed (verify sync, graph pads, flashinfer workspace,
rank-0 draft-pick broadcast, fp8 dequant pairing). GDN prefill beyond ~109
tokens is upstream-nondeterministic — byte gates only on short outputs. Canon:
no A/B without a same-boot A-vs-A floor; first boot after cache changes is a
JIT outlier; floor scales with measurement length. Determined-answer probes:
underdetermined text can only report "different", never "wrong".

## 11. Device identity (order trap)
torch order != NVML order on multi-vendor-generation rigs. The ONLY bridge is
the IdentityMap (registry/nvml.py) keyed by UUID/PCI-BDF. Never feed a CUDA
ordinal to NVML or vice versa; never key caches on the masked CUDA view.
This also covers the custom-group object exchange: it names the world gloo
cpu_group instead of letting torch pick a staging device.

## 12. Robustness canon
Rank-local condition BEFORE any group collective (hang family); bounded waits
with fixed pool universe; bounded peer-liveness instead of endless spin;
ColdBuild error unmasking (never substitute "lower mem-fraction" for a real
error); quant guards fail loudly instead of silently downgrading; JIT cache
poisoning family (stale batons, foreign-worktree kernels, cold-JIT =
capture-cost illusion).

## 13. Serving surface
OpenAI-compatible with `--reasoning-parser qwen3 --tool-call-parser
qwen3_coder` (server-side fix, no template patches); fast lane, priority
scheduling, admission throttle, prefill delayer; training tenant + idle
workbench (ledger + pause rung); `/session_handover`; `/kv_reshard`.

## 14. Dashboard
Guided config wizard with honest refusals, comm benchmark suite with
anonymization gate, energy metering (tok/s + J/token), benchmark tiles with
measured/estimate/absent provenance, one-click knee-point probe, self-update
with auto-rollback, GitHub result posting (opt-in PAT).

## 15. Model bring-ups (boot-proven)
Qwen3.5/3.6 family (all quants), Gemma4 26/31B (+GGUF, quadratic-mask skip),
Llama family, Mistral Small 24B FP8 + ministral3 SWA fix, Deckard-40B/Tess-27B,
122B-A10B offloaded, 35B-A3B, DeepSeek-V4-Flash-0731 GGUF TP=3 offloaded with
OWN sm86+sm120 attention paths (e4m3 bit-decode, f32 staging, indexer arch
dispatch, mask-oracle fix). Nemotron-Puzzle class structurally covered,
unbooted.

## 16. Measurement / window infrastructure
gpu-arb (UUID-based holder + heartbeat — stop the heartbeat BEFORE releasing),
forward_peak.py (VRAM corridor judged AT PEAK, not idle), cachetrim with
--ready-url self-retirement, expert_stats (router distribution + hit rate),
CollectiveClock (compute vs wait per rank), measured-KV-budget stale-boot trap.
