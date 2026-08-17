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
  (per-path meaning below — do not read this as "experts between ranks" in
  general), `--rank-moe-resident-fraction` (GPU/host split WITHIN a rank),
  `--rank-kv-ratio` (`coupled|speed|vector` — decouples KV split from weight
  split), `--rank-auto-reserve-mib`, `--rank-gpu-memory-mib` (absolute
  per-rank MiB budget with a line-item ledger incl. lane pools).
  Read `--rank-moe-ratio` precisely: under the **#82 GGUF expert-dim shard** it
  moves whole experts and therefore the COMPUTE assignment (owner runs the
  expert, foreign ids remap to a zero pad, the TP all-reduce sums the disjoint
  partials); on every other MoE path it splits the expert INTERMEDIATE dim, so
  every rank still computes every routed expert and only the weight slice
  moves. `--rank-moe-ratio link` (#394 slice 3) solves the vector instead of
  taking it: the GPU-resident expert mass stays exactly where the base plan put
  it (VRAM-neutral) and the STREAMED remainder is apportioned by the measured
  link weights, which equalises the per-rank transfer time the group waits on.
  Optional per-rank cold-traffic coefficients
  (`SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS`, measured from a prior boot's #390
  dump) replace the first-order "a cold expert is fetched, a resident one is
  not" model; without them the solve labels itself UNCALIBRATED. Refused by
  name when offload is off, when the link provenance is `absent`, or under
  `ep_size>1`. Resolved ONCE in the launcher — a symbolic value that reaches a
  worker is a hard error there, never a silent fall back to the base plan.
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
**The fundability gate prices the vector the boot runs (#437).** A FIXED KV
token vector keeps the relative base-plan pricing; a MATCHED one
(`--rank-kv-ratio capacity|speed`, and the phase arms since #435) has no
unused capacity to price, so every rank is checked ABSOLUTELY against the
derived reserve demand on ALL cards. Before #437 `capacity` mode accepted
16,1,1 at reserve 3000,2700,2700 -- #264's OOM config -- because it compared
a matched residual against an identical matched base; the capacity-directed
objective did not consult the gate at all. #330's 400 MiB corridor is priced
alongside the demand and REPORTED (`CORRIDOR-TIGHT`), never binding
(`SGLANG_PLANNER_CORRIDOR_MIB` overrides it; the number itself lives once in
`registry/ledger.py`).
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
- **#394 cold-shard chain** (slices 1+2 merged): measured H2D provenance chain
  (env > card-probe > nvml-negotiated > refusal; `absent` unselectable),
  `cold_tier_shm.py` shared-DRAM segments (UUID/BDF identity, manifest read
  lazily after load, header sealed last, PROT_READ views with kernel-enforced
  write protection). **Slice 2 wires the fetch path** (`cold_tier_fetch.py`):
  a rank-uniform owner map derived from the same `partition_cold_experts` the
  staging plan uses (plan `digest()` pins the uniformity), the cold pool
  ALLOCATED IN the segment rather than copied into it, and
  `MoEExpertOffloadCache._fetch` sourcing a delegated expert from the owner's
  `PROT_READ` view over this rank's own link. Behind
  `SGLANG_MOE_COLD_TIER_SHM=1`; with it off the slice-1 boot refusal for
  delegation on disjoint expert shards is unchanged, field for field.
  **Honest scope of slice 2**: byte ownership moves, COMPUTE does not, so
  per-rank H2D is predicted unchanged.
  **Slice 3 (#439) moves the compute assignment** and is where ANALYSE_393's
  Path A′ lives. It needed no new mechanism: the #82 expert range IS the "moe"
  family vector, so the slice is a SOLVE plus its wiring
  (`layers/moe/expert_compute_placement.py`, `--rank-moe-ratio link`, see §1).
  Predicted on the reference recipe from the 2026-08-02 battery's own measured
  inputs: clock rank 858 s → 632 s (1.358x) uncalibrated, → 542 s (1.584x) with
  the coefficients calibrated off the equal arm, against BENCH_394's 1.536x
  ideal-placement reference. All three are PREDICTIONS; nothing has booted.
  BOOT-PENDING: `scripts/dev/394_s2_proof/` (eager arms 1+2, plus the slice-3
  arms `ARM=compute` / `ARM=compute-cal` specified in `ARM3_COMPUTE.md`), and
  the graph seam,
  which refuses by name until the UVA pointer for a `cudaHostRegister`'d peer
  mapping is verified (`SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1`). Graphs incl.
  CPU-MoE remain IMPLEMENTATION EFFORT, not blocked: UVA reads, cudaGraph host
  nodes, CUDA>=12.4 conditional nodes, and graphs pin ADDRESSES not CONTENTS
  (spill/restore under fixed buffers is legal).
- **HiCache** L1-L3 prefix cache (validated with uneven DCP/TP; storage key
  includes kv-dtype; runtime attach/detach works on UnifiedRadixCache). The
  L2 host tier's `page_first_direct` transfer path was blocked on this rig by
  a segfault in `transfer_kv_all_layer_direct_lf_pf` (#436, cu12/cu13
  `cudaMemcpyBatchAsync` ABI split); unblocked by the cu13 `sgl_kernel`
  rebuild.
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
  (audit #421 + #410): exactly ONE production consumer — #410's session
  checkpoints, through `memtier/consumers.py` (`checkpoint_tier_targets`, the
  write-path selection helper, memtier cut 3 credit). "All consumers pick
  targets from it" remains the TARGET rule: the PRE-EXISTING offload/spill
  paths (#286 register, #394 cold tier) still carry their own target lists,
  and migrating them is cuts 4/5. The #421 pin test
  (`test_unwired_features_421.py`) no longer asserts "zero consumers"; it pins
  the #410 CALL SITE, so an unreached registry cannot come back unnoticed.
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

**Server-side conversation checkpoints, branching and rewind** (#410) reuse
that same export: a checkpoint IS a handover snapshot whose destination is a
storage TIER instead of a peer group, so there is exactly ONE session
serialization in the fork (the versioned #261 manifest, additively extended
with a `checkpoint` envelope; the task ledger also names it as #411's
portable-session format). `POST /session/{id}/checkpoint` freezes KV pages AND
the GDN blob (#212 gate inherited, keyed on `supports_mamba()`);
`/session/{id}/branch` opens a new session from it and `/session/{id}/rewind`
moves an existing one back, both WITHOUT re-prefilling. Branching copies
nothing — the radix tree already shares a common prefix and splits at the
divergence point, so #410 adds only the `inc_lock_ref` pin there, and the
reported accounting asserts `copied_pages == 0`.
**TWO TIERS OF PIN, and they are not interchangeable.** `inc_lock_ref` protects
the radix chain IN MEMORY for the life of the process; the referenced pages
still sit in the HiCache file store as ordinary LRU entries, so a checkpoint
with only that pin survives radix eviction and does NOT survive file-tier
eviction or a restart. `take_file_tier_pins`
(`managers/session_checkpoint.py`) takes the second tier through the #410 pin
ledger (`mem_cache/pin_ledger.py`, honoured by `LRUFileEvictor`), and raises
`PinCoverageIncomplete` naming the references it could not pin -- answering "are
this checkpoint's pages on the file tier at all" at CHECKPOINT time rather than
letting the shortfall surface at the branch. A checkpoint placed by #407 on
vram/host has no file tier and is logged as unprotected, not refused.
The file evictor charges **ALLOCATED** bytes (`max(st_blocks*512, st_size)`),
the same unit the pin ledger charges, so `reclaimable = used - pinned` is a
coherent subtraction; `stats()` reports `accounting_overshoot_bytes` if the two
ever diverge again. The tier comes from the #407 registry
(VRAM → RAM → Disk by age/durability, provenance-labelled, named refusal when
nothing is admissible). Restore is #261's `verify_import` (#241 identity) plus
a geometry gate; cross-geometry is a NAMED refusal pointing at the offline
umsharder, never a silent conversion. Same v1 limits as #261: TP=1/PP=1,
`page_size == 1`, `file` backend; plus `--hicache-mem-layout page_head`
refused by name, because its host write-back is the `lf_ph` route that
segfaults (#441a) and a checkpoint writes a whole session through it.
Behind `--enable-session-checkpoints`
(default off). BOOT-PENDING: the byte gate in
`docs/dev/DESIGN_410_session_checkpoints.md` §8 (resume trajectory vs
never-paused reference, branch-then-continue vs fresh-prefill,
parent-untouched) has NOT been run — nothing in #410 has executed on a GPU.

Also wired on the tip but easy to miss (audit #421): the regime-controller
gate machinery, KV-pressure rung-dependency refusals, the hibernate flag
contract (`hibernate_dir` + weights/draft CPU/disk backup flags), and a
118-name retired-env guard that refuses stale SGLANG_* variables loudly.

## 7. Collectives / transport
**barlink p2p: HOST has it, BAR1 does not.** `barlink_host.send`/`recv`
(`barlink_host.py:1100`, `:1120`) is a working point-to-point path with a
per-pair slot, flags, per-peer sequence and bounded timeout. BAR1 has no
equivalent: its three kernels are collectives and it owns no p2p kernel.
`barlink_bar1_p2p.py` supplies the DESK half of that seam — directed-pair
slot algebra, 256-byte flag lines, append-only layout (`off_p2p = -1` when
absent), caller-side chunking, and refusals carrying their arithmetic —
wired into nothing, so existing layouts stay byte-for-byte. CAPTURE: send is
capturable (`put()` is a stream `memcpy_async`), recv is NOT (no device-side
wait; a host spin in a capture raises `cudaErrorStreamCaptureUnsupported`),
so a PP crossing over this seam is BREAKABLE and priced with #494's clock.
`NOTE_732_bar1_p2p_seam.md`.
**barlink is COLLECTIVE-ONLY, and that is load-bearing for placement (#732).**
Its dispatch seams are `all_reduce` (`parallel_state.py:1100`),
`reduce_scatter*` (`:1299`, `:1374`, `:1498`) and `all_to_all_single*`
(`:1438`, `:1450`, `:1480`) — there is no `send`/`recv` on `barlink_bar1.py`.
So BAR1 accelerates COLLECTIVES (measured 1.13/1.34/1.15/1.04/1.30x vs NCCL at
20 KiB…16 MiB, `DESIGN_407_memory_tier_registry.md:131`) and carries no
point-to-point traffic. A PP stage handoff is point-to-point
(`send_tensor_dict`, `:2178`, on a `use_custom_allreduce=False` group `:3121`),
so "no P2P" reasoning about PP crossings must NOT be read as "no direct
transport" — and equally, BAR1 must not be dismissed for collective work.
The measured BAR1 row's own citation is WRONG: `DESIGN_407:131` credits
`EVAL_gdr_uebernahme.md:141`, which is a dmabuf-GPU-RDMA-over-RoCE document
with zero matches for those numbers; the true source is
`FEATURES_VS_UPSTREAM.md:1341` + commit `137e3a6c25`. BAR1 is also NOT a
uniform win: on the fast x8 PAIR it loses 1-8 MiB, down to 0.81x
(`FEATURES_VS_UPSTREAM.md:1349`), so 3-rank ratios must not be reused for a
2-rank group. Consequence for #705's TP-decode family split: its baseline `ar_10kb_us`
31.0–33.7 µs is an **NCCL** probe (`uneven_perf.py:1329-1330`), and re-scaling
by the measured BAR1 ratio turns its +0.022…+0.152 ms margin NEGATIVE
(−0.034…−0.356 ms) — a faster interconnect makes collective-REMOVAL worth
less, so that refusal strengthens. `ANALYSE_732_bar1_repricing.md`.
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
wedged run). **Scoped slow-boot warning**: barlink BAR1 × uneven weighted DCP × an
fp8-quantized checkpoint warns loudly at ModelRunner boot instead of refusing
(#438a). What #424 recorded as a wedge is a slow FIRST boot: on a cold JIT
kernel cache the first CUDA-graph capture batch spends ~190 s per rank
(184-197 s, three ranks concurrently) inside the JIT cold-build window, and
under the raw ~30 s BAR1 cap the peers' spin kernels tripped their deadline
about six times over inside it — which is the "~30-40 s per collective"
crawl. Two proofs, both 2026-08-02: capture (`.../2026-08-02_431_recheck/`,
12/12 in 4:58, READY after 6:05, 176/176 requests, no
`Bar1CollectiveAborted`) and full serving load
(`.../2026-08-02_435_coupling_fp8bar1/`, both FP8 layouts, 9× `ACHIEVED=bar1`
per arm, full probe sets, no abort/PeerLost/CollectiveTimeout in any log).
Warm boots are normal speed. Restore the old hard refusal with
`SGLANG_BARLINK_REFUSE_FP8_UNEVEN_DCP_BAR1=1`; the legacy
`SGLANG_BARLINK_ALLOW_FP8_UNEVEN_DCP_BAR1` is kept and is not a no-op — `=0`
still means "do not admit this arm" and still refuses, `=1` is still honoured
but now redundant and says so. INT8-W8A8 over BAR1 and fp8 over NCCL remain
untouched. See `docs/dev/ANALYSE_431_fp8_bar1_dcp_deadlock.md`. Still
unmeasured: a boot from a genuinely empty `extcache_docker`.
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
workbench (ledger + pause rung); `/session_handover`; `/kv_reshard`;
`/session/{id}/checkpoint|branch|rewind|checkpoints` (#410).

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
