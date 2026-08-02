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
  sm75). Slice 3 merged and cross-rig pp=2 validated: world-MIN
  `max_total_num_tokens` before the reduce, `--pp-stage-ratio`
  (score-proportional, snaps to full-attention boundaries), stage-local mamba
  slots, `auto` under PP with an agreement gate, `SGLANG_PP_SHAPE_CACHE` cuts
  boundary-send by −9.8/−9.2 % at bs=1 (0-1 % floor otherwise) — note the
  in-server counter reads 249 µs, which is not the standalone wire-transfer
  figure.
- **TP5+ emulation** via NCCL multi-rank co-location (several ranks per card).

## 2. Planner / solver
Key solver: water-filling over an affine cost model, pair-matrix collective
term, roles/nesting as box bounds, Pareto+knee, admissibility gates,
`coresident_budgets()`. Measured phase optima on the reference rig (1x RTX
5090 + 2x RTX 3080, Qwen3.6-27B, ctx 32768; RIG EXAMPLE, not a portable
default): prefill 10,1,1 (+ decoupled KV 2,11,10 keeps capacity), decode
~3,2,2 — solve your own via `--rank-perf-tune phase-prefill|phase-decode`
and read the `CHOSEN` vector off your boot's log. Under `--rank-perf-tune phase-*` the
solve now also OWNS the coupled KV token vector (#435): the chosen candidate's
matched `predict_capacity` vector is seeded into the boot instead of the
VRAM-budget split, so the pool the runtime sizes is the one the admissibility
gate accepted (#433 measured the gap: 125 504 vs a predicted 358 693 tokens).
An explicit `--rank-kv-ratio` still wins; the hand-paired
`--rank-mlp-ratio X + --rank-kv-ratio Y` of #354/#424 is no longer needed.
`--objective energy` end to end with refusal over silent substitution. `planner/rejected.py` = machine-readable
register of discarded approaches — check it before re-proposing anything.

**Generality (#434 slice 1).** Plain `--rank-tp-ratio auto` is the documented
CAPACITY-FIRST default (byte-proportional to the VRAM budgets, no probe); it
now names the per-task optimizer and the flag that engages it in the CLI help
and in one boot log line, and calls out a hand-pinned `--rank-mlp-ratio` as the
solution of some earlier operating point. `--rank-perf-tune dec` no longer
returns the base split on the strength of M22's reference-rig "decode is flat"
finding: it SOLVES the bs=1 decode round time from the rig's own effective
bandwidth, and reports flatness as a result when that is what the profile says.
Every objective therefore solves from per-(rank, family) profile scores.
Constant audit: `docs/dev/AUDIT_434_planner_constants.md` (62 classified;
19 RIG-FITTED, 16 named follow-ups FU-434-1..16). The cost model now prints
which calibration scalars are BORROWED from the development rig rather than
only which were overridden. Standing hermetic proof suites on synthetic
foreign rigs: `test/registered/unit/planner/test_planner_generality_434.py`
(profile-follows, symmetry-has-no-lever, relabeling/scale/name invariance,
AST leak guard) and `test_borrowed_calibration_434.py` (a measurement may only
be applied to hardware it matches). Probe-first bootstrap on unknown hardware
is designed, not built: `docs/dev/DESIGN_434_probe_first_bootstrap.md`.

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
  50s→8-14s) + suspend-to-RAM (memory saver; reaches the legacy hybrid-SWA
  `SWAKVPool` since upstream #32213 — before that it was silently a no-op
  there, while `UnifiedSWAKVPool` already honoured it).
- **Runtime VRAM dial** per card (VMM page return), **KV pressure ladder**
  (geometry stages instead of rejects; explicit ladders work; rung-dependency
  refusals exist and fire). `--kv-pressure-ladder auto` mode wired via
  rig-profile bridge (#428), boot validation pending — the table is
  computed from the rig profile by the #272 planner, rank-uniformly and
  UUID-keyed, and inventories only rungs whose actuator this configuration
  wires. Capacities are labelled placeholders until the measured figures
  arrive. BOOT-PENDING: `scripts/dev/428_boot_checks/`. **KV resharding**
  at phase boundaries (delta move <1 s, `kv_reshard_vectors`), **GDN slot
  ladder** (resident-state cap + idle vacate → VRAM back to KV pool).
  `--lane-offload-profile/-class-policy/-park-targets` are wired at runner
  init once-per-process (#428), boot validation pending; a typo now refuses
  there too. The park chain reaches the register and the movement layer's
  default reads it — but nothing in
  production constructs the movement backend yet, so the chain has a consumer
  PATH, not a consumer. Whole surface is behind `SGLANG_OFFLOAD_REGISTER=1`
  (dark launch). BOOT-PENDING: `scripts/dev/428_boot_checks/`.
- **memtier registry**: tier ids with volatility + payload class and
  provenance `measured|estimate|absent` (absent refuses use). HONEST STATE
  (audit #421): ZERO consumers wired today — "all consumers pick targets from
  it" is the TARGET rule, not the current state; existing offload/spill paths
  still carry their own target lists. The #421 pin test
  (`test_unwired_features_421.py`) enforces that statement and must be updated
  in the same merge as the first real consumer.
  Slice 1b (#407 / directive #434) made it hardware-general:
  `TierRegistry.for_machine()` fingerprints the box from NVML UUIDs (#397
  canon), applies a stored profile ONLY at the scope its hardware match
  licenses (`EXACT` = every tier; `MODEL` = card templates only, no host /
  filesystem / remote row), and otherwise bootstraps from live facts with
  measured sizes and every cost ABSENT naming its probe. `from_profile()` no
  longer defaults to the bundled rig profile — that default handed one
  development box's host RAM, ZFS pool and 40G peer to every machine.
  Measurements are ingested from the EXISTING artifacts (`card_probe` #213,
  rig artifact #271, `capability_matrix` #278) by `memtier/adapters.py`;
  #407 adds no probe of its own. `TierTransport.link_path` +
  `link_disjointness()` expose PATH identity for #423's striping gate, with
  `DISJOINT` requiring complete paths on both sides and `UNKNOWN` being a
  refusal. Design: `docs/dev/DESIGN_407_memtier_registry.md`.

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
re-sharder as its own spec type: MERGED and GPU-gate-passed
(`POST /session_handover`, five-phase at session scope, hard GDN-blob gate
keyed on `BasePrefixCache.supports_mamba()`; proven byte-identical to a
never-moved reference via a real cached-tokens import, `cached_tokens=1152`
on resume, plus seven named-refusal negative controls) — the declared v1
limit stands unchanged: a booted TP>1 destination still needs the offline
manifest-scoped umsharder (`page_size == 1`, inherited from
`dcp_owner_mode`) to reshape into its geometry first, live handover does not
do that reshape in-process.

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
**Collective-decision recorder** (`barlink_uniformity.py`, #431): per-rank
ordered log of every `(op, nbytes, path, rounds)` dispatch decision plus a
pure `first_divergence` comparator — the standing instrument for the
rank-local-condition-before-a-group-collective family (#94/#194/#312/#431).
Off by default (`SGLANG_BARLINK_RECORD_DECISIONS=1`, optional per-rank
on-disk dump via `SGLANG_BARLINK_RECORD_DUMP_DIR` for post-mortems on a
wedged run). **Scoped refusal**: barlink BAR1 × uneven weighted DCP × an
fp8-quantized checkpoint is refused at ModelRunner boot (#424 evidence;
INT8-W8A8 over BAR1 and fp8 over NCCL are untouched). Override for the repro
window: `SGLANG_BARLINK_ALLOW_FP8_UNEVEN_DCP_BAR1=1`. See
`docs/dev/ANALYSE_431_fp8_bar1_dcp_deadlock.md` — GPU proof still pending.
**BAR1 deadline + loud abort** (#431 fix slice): the three BAR1 kernel launch
sites go through `resolve_timeout_cycles`, so the documented 40x JIT
cold-build extension finally reaches the one transport whose kernels spin on
a device deadline (identity outside the window, so serving and the captured
graph are unchanged). A tripped spin kernel now raises
`Bar1CollectiveAborted` with rank/op/rounds instead of continuing over a
partially written buffer — checked after every host-path collective and, for
captured decode, at the CUDA-graph replay boundary
(`barlink_abort_gate.py`); never inside a stream capture, where the device
read would be illegal. Knobs:
`SGLANG_BARLINK_BAR1_ABORT_CHECK=0` (restore the old silence),
`..._CHECK_EVERY=N`, `..._CHECK_REPLAY=0`.

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
capture-cost illusion); reference-twin drift family (#418 #425 #427 -- a torch
reference that disagrees with the kernel it validates, hidden by an oracle
that compares only the region where they agree; fix the reference AND widen
the comparison, and pin whether the reference is reachable from serving).

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
Qwen3.5/3.6 family (all quants), Gemma4 26/31B (+GGUF, quadratic-mask skip;
Gemma3RMSNorm runs the fused sgl-kernel path for 2-D and high-rank inputs,
adopted from upstream #32670 — do not re-add an eager-only forward_cuda),
Llama family, Mistral Small 24B FP8 + ministral3 SWA fix, Deckard-40B/Tess-27B,
122B-A10B offloaded, 35B-A3B, DeepSeek-V4-Flash-0731 GGUF TP=3 offloaded with
OWN sm86+sm120 attention paths (e4m3 bit-decode, f32 staging, indexer arch
dispatch, torch/triton reference-twin parity: indexer mask oracle, SWA
page-index wrap oracle, page-table rounding, top-k seq_len contract).
Nemotron-Puzzle class structurally covered, unbooted.

## 16. Measurement / window infrastructure
gpu-arb (UUID-based holder + heartbeat — stop the heartbeat BEFORE releasing),
forward_peak.py (VRAM corridor judged AT PEAK, not idle), cachetrim with
--ready-url self-retirement, expert_stats (router distribution + hit rate),
CollectiveClock (compute vs wait per rank), measured-KV-budget stale-boot trap.
