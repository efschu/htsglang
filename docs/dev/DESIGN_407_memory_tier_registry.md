# DESIGN #407 — one memory-tier registry, many spill consumers

Charter (user directive, verbatim): *every memory you have access to must be a
"level cache", a spill target or an offload target depending on volatility —
disk / RAM / VRAM, local as well as remote.* #224 is consumer number one, #305
is the policy layer above.

> **READ THIS FIRST — there is a second #407 design document, and it wins where
> the two disagree.** `DESIGN_407_memtier_registry.md` (2026-08-02, slice 1b,
> directive #434) is scoped to what #434 changed and to slice 1. **This document
> remains the design of record** for the node layer, the consumer survey (§2),
> the tier interface (§3), the measurement plan (§4) and the cut plan (§5) --
> the newer one cites those rather than restating them. But it overrides this
> document on exactly three points, enumerated in its §2: the profile default
> (`profiles/rig1.json` is a *candidate* matched against hardware, not a
> normal-path default), the empty `cards` list in that profile (functional, not
> a TODO -- it can only match at MODEL scope and asserts nothing about a
> reader's host or disks), and `TierCaps` provenance (values may now enter from
> an artifact adapter, though `apply_outcome` is still the only writer and its
> refusals are unchanged). Everything else here stands, including all four
> exclusions (X1 HiCache ladder, X2 GDN state, X3 cross-rig GPU-to-GPU, X4
> compute placement) and the C1/C2 contradictions.
>
> This pointer was added on 2026-08-17. The forward reference already existed;
> the back reference did not, so a reader who opened this document first had no
> way to learn it had been partially superseded. See
> `VERDICT_407_two_designs.md`.

Desk work, no cards (`CUDA_VISIBLE_DEVICES=99`). Branch
`docs/407-tier-registry-design`, base `1421d20dce`
(`integration/r3-probe-next2`). This document designs; it changes no code.

---

## 0. Verdict

**#407 is not a new abstraction. It is a NODE layer plus a reconciliation.**

The survey (§2) found that the expensive half of a memory-tier registry — *what
does it cost to move a byte from here to there* — **already exists four times
over**, each with its own reader, its own provenance convention and its own
absent-value rule:

| # | Edge source | Keyspace | Owner |
|---|---|---|---|
| E1 | `planner/cost_model.py` `PairMatrix` / `Hop` | card UUID x card UUID, directed | #348b |
| E2 | `distributed/device_communicators/barlink_path_rates.py` `PathProfile` | named transport route, affine `base_ms + per_byte_ms` | #279 |
| E3 | `model_executor/offload_movement.py` `PeerPathCapability` / `PeerProbe` | **CUDA ordinal** x CUDA ordinal, directed | #286 |
| E4 | `rigmon/card_probe.py` `h2d_gbs` / `d2h_gbs` | card UUID, one direction to host | #271 |

E1, E2 and E3 each carry an aperture and a bandwidth for a directed pair. E2
and E3 both want the *effective* (not nominal) BAR1 aperture out of
`scripts/p2p_readiness/capability_matrix.json`; only E2 has a loader for it.
Nothing reconciles them. This is exactly the condition #348b found among the
three placement planners and fixed for compute rates — repeated one layer down,
for bytes at rest.

What does **not** exist anywhere is the node layer: an enumeration of the
memories themselves, with a stable identity, a declared capacity, and a rule
about which kinds of object each may hold. In its place the tree carries **four
disjoint tier vocabularies**, none of which knows about the others:

| Vocabulary | Verbatim | Site | Covers |
|---|---|---|---|
| V1 park-target ladder | `PARK_TARGETS = ("peer_vram", "host_ram", "remote")`, `OWN_VRAM_TIER = "own_vram"` | `offload_register.py:132-133` | #286 items |
| V2 spill destination chain | `LOCAL_DESTINATION = "local"`, `SUPPORTED_PARK_BACKENDS = ("file", "mooncake", "dynamic")` | `kv_session_spill_destination.py:101-106` | #224 sessions |
| V3 storage medium | `StorageMedium{GPU, CPU_PINNED, DISK, EXTERNAL}` | `disaggregation/kv_events.py:82` | telemetry only, never placement |
| V4 HiCache backend names | a closed `choices=[...]` list of nine | `server_args.py:3775-3790` | L3 blob stores |

Expert offload (#77/#123) — the largest byte mover in the fork — appears in
none of them: its tier is the string literal `device="cpu"` plus
`.pin_memory()`, written out five times (§2.1).

Three consequences shape the whole design.

1. **The registry must subsume, not add.** `DESIGN_305_multi_model_serving.md`
   §6 already forbids the obvious mistake, by absolute line reference:
   > **One ledger. No second accounting.** The residency ladder must express a
   > model's rung as posts in that ledger, exactly as the offload register
   > expresses parked items — otherwise two allocators will each believe they
   > own the same bytes, which is the failure the ledger was introduced to
   > prevent.

   There are already **two** capacity ledgers: `registry/ledger.py`
   (cross-process, one JSON file per card, keyed by NVML UUID, enforcing
   `sum(reserved) + corridor_bytes(C) <= nvml_total_bytes(C)` with #330's
   400 MiB corridor) and `offload_movement.py:514` `CapacityLedger`
   (in-process, keyed by `(target_name, cuda_ordinal)`). #407 makes the second
   a *view* of the first. It does not write a third.

2. **The registry answers what exists and what it costs; never what to do.**
   Policy lives in #305 (`registry/rungs.py`, `registry/arbiter.py`) and #363
   (`planner/lever_profiles.py`). The registry's own query returns an *ordered
   candidate list with its ordering key exposed*, so a policy layer can reorder
   it. It never picks.

3. **Three numbers this design would have liked to lean on are not measured.**
   The peer-VRAM "1-3 us posted write" class is an explicit *assumption*
   (`EVAL_p2p_prefill_decode_split.md:140`, under "Tragende Annahmen"); NVMe
   latency does not exist in the tree in any unit; host DRAM bandwidth is
   derived from an *assumed* DDR4-3200 peak because the VM hides SPD. Under
   #348b's rule — measured, estimate, or a named absence, and no fourth case —
   the registry ships those three as `ABSENT` and `ESTIMATE`, and §4 says who
   measures them.

**The two contradictions the design had to bend around** are stated up front
because both are shipped, in-tree, and mutually exclusive:

* **C1 — two opposite total orders over the same axis.** #224 enforces, as a
  validation error, that `local` (pinned host RAM) must be the FIRST
  destination and every further tier is *below* it
  (`destinations_error`, `kv_session_spill_destination.py:146`). #286's ladder
  puts `peer_vram` *above* `host_ram`
  (`DEFAULT_PARK_TARGET_ORDER`, `offload_movement.py:116`, ladder at `:687`).
  Both are correct for their own tier set and neither generalises. §3.5
  resolves it: the ordering is not a constant, it is a query over the
  *staging graph*, and "local first" is a reachability fact about blob-store
  tiers, not a bandwidth law.

* **C2 — the one existing tier ledger is positionally keyed.** `CapacityLedger`
  keys peer-VRAM budgets by `device: Optional[int]`, a CUDA ordinal, and
  `budgeted_peer_devices()` returns `List[int]`. Under `AUDIT_331` that is
  **SCOPED** (legitimate) as long as it never crosses a process boundary — and
  it is the moment #305 §6 gets its wish. On this rig the 5090 is CUDA ordinal
  0 and NVML index 1, so the failure mode is silent and card-swapping. §3.1
  makes tier identity UUID/BDF-derived and gives the ordinal a one-call,
  in-memory resolution site.

Cut 1 (§5) is S: a module, a read-only enumeration, one dashboard row, zero
consumer changes. The highest-yield early cut is not #224 — it is publishing
the rank-to-card-UUID vector that #394's already-merged link-proportional
sharding is currently *starved* of (§2.1, §5 cut 2): a 145 -> 86 ms/token bound
on the cold tier, code already written and tested, blocked only on a datum a
registry has anyway.

---

## 1. Tier inventory

### 1.1 What is on this rig, with provenance

Provenance follows `cost_model.Provenance`: **MEASURED** came off a probe on
this box, **ESTIMATE** came out of a formula over measured inputs, **ABSENT**
has no value and carries the reason. There is no fourth case, and no number
below was supplied from general knowledge.

| Tier | Capacity | Bandwidth | Latency | Provenance / source |
|---|---|---|---|---|
| **T0 local VRAM** — RTX 5090 | 32607 MiB total, 31.34 GiB after context | membw 1558 GB/s, gemv 1533.5 GB/s | — | MEASURED. `rig-runbook.md:2977,3468`; `TASK_103_SPEC_K_POLICY.md:187`; `INTEGRATION_R3_VALIDATION.md:12152`. Probe: `rigmon/card_probe.py` |
| **T0 local VRAM** — RTX 3080 x2 | 20480 MiB each | membw 723 GB/s, gemv 718.2 GB/s | — | MEASURED, same sources |
| | aggregate 72 GB over three cards | | | `ANALYSE_389_nvme_expert_tier.md:99` |
| **T1 peer VRAM** via barlink BAR1 | BAR1 aperture: 3080 **256 MiB** nominal, of which **96 MiB maps contiguously**; 5090 **32 GiB** (ReBAR) | vs NCCL, interleaved, bf16, 3 cards: **1.13 / 1.34 / 1.15 / 1.04 / 1.30x** at 20 KiB / 80 KiB / 1 MiB / 4 MiB / 16 MiB | smallest datapoint is a **20 KiB three-rank all_reduce at 45.59 us** — a collective, not a point latency | MEASURED. Commit `137e3a6c25` ("BAR1-Transport laeuft: drei Raenge 1,03x bis 1,51") is the source VERIFIED PRESENT on this branch; `FEATURES_VS_UPSTREAM.md:1341` carries the table but is not tracked here. The former `EVAL_gdr_uebernahme.md:141` citation is REMOVED (#732 amendment, re-verified 2026-08-17): that file exists at `docs/EVAL_gdr_uebernahme.md` and contains none of these figures -- `1.13`, `1.34` and `45.59` all return zero matches in it. |
| | **effective** aperture per directed pair | | **"1-3 us posted write"** | **ABSENT** — `scripts/p2p_readiness/` has never been run, no `results/` exists. RE-VERIFIED 2026-08-17 against the #732 amendment, which reported this claim stale: on this branch it HOLDS. The package is present (`capability_matrix.py`, `d2d_bench.py`, `nccl_transport_check.py`, `p2p_common.py`) and there is no `results/` directory in any checkout examined (`wt-train2b`, `shvllm`, `wt-706-hicache-keys`, `htsglang`). A `results/` seen elsewhere would be untracked and local to that worktree. The us class is an *assumption*, `EVAL_p2p_prefill_decode_split.md:140` |
| **T2 host RAM** | 103.3 GB `MemTotal`, cgroup limit 98.5 GiB, no swap | DRAM sustained **32-45 GB/s, central 38** | — | **ESTIMATE.** Peak 51.2 GB/s is *assumed* (dual-channel DDR4-3200; `dmidecode -t 17` empty under the VM), `ANALYSE_393_ik_llama.md:299-300`, named as an open item at `:554` |
| **T2 host RAM** — per-card reach | | H2D pinned: **6.4 / 13 / 13 GB/s** (gen4 x4 / x8 / x8), aggregate 32.4 GB/s. Second measurement, barlink host-staged: 14.3 / 6.5 / 13.2 GB/s | — | MEASURED. `ANALYSE_393_ik_llama.md:301-304`; `rig-runbook.md:98`; `pd_disagg_single_node.md:52-53`. Probe: 64 MiB pinned, one direction alone, best-of wall clock, `card_probe.py:561-586` |
| **T3 local NVMe** (`/spinning`, ZFS) | 729 GB free | **1.8 GB/s cold**, `iflag=direct`, reproduced three times. Warm 3.8 / 9.5 GB/s are ARC, explicitly not credited to the tier | — | MEASURED. `ANALYSE_389_nvme_expert_tier.md:77-89`; commit `71fc6356be` |
| | | | any storage latency figure | **ABSENT** — no `fio`, no `iodepth`, no per-read us anywhere in the tree. All #389 arithmetic is bandwidth-only |
| **T4 remote rig-2 RAM** over 40G RoCE | rig-2 = RTX 2080 Ti host | NCCL-over-sockets **2.07 GB/s**; staged RDMA 1 MiB **2.83 GB/s**; iperf3 single stream 1930 MB/s. (100G line present but slot-limited to 3.43 GB/s, PCIe-3.0-x4-bound) | 8 B message **1.47 us** (40G) / 1.58 us (100G); UCX barrier world 2 **8.30 us**; all_reduce 8 KiB **44.92 us**, 64 KiB 55.79, 512 KiB 237.70 | MEASURED. `rig-runbook.md:599,1264,1472`; `EVAL_gdr_uebernahme.md:125`; `FEATURES_VS_UPSTREAM.md:680-700` |
| **T4 remote rig-2 VRAM** (GDR direct) | | direct GDR into a 3080 BAR at 1 MiB **0.83 GB/s** vs staged 2.83 — but #278's retake overturns the size story: no crossover point, a **BAND at 64-80 KiB**, direct wins outside it (20 KiB 1.26x, 1 MiB 1.19x, 4 MiB 1.24x), **BAR size irrelevant** | 8 B direct 4.99 -> staged 3.77 us | MEASURED. `EVAL_gdr_uebernahme.md:124`; `INTEGRATION_R3_VALIDATION.md:6176-6210` |
| | serving-level cross-rig collectives | | decode all_reduce **~89 us**, verify all_reduce **~202 us**; host staging carries ~86% of it | MEASURED, #266 verdict, `INTEGRATION_R3_VALIDATION.md:4958-4960` |
| **T5 remote disk** (rig-2) | | | | **ABSENT** — never probed. Reachable only through T4's wire; §4 records it as a declared-but-unmeasured tier |

Two facts about T4 that the registry must carry as *properties*, not footnotes:

* **The NIC relay serialises.** Three pairs in parallel: 2.34x latency, only
  1.28x aggregate — the single Gen3-x4 NIC is the wall (#278 V5). A remote tier
  is not width-scalable on this rig.
* **GPU-to-GPU across the rig boundary was never demonstrated** — and that is
  precisely the NORDSTERN shape (`EVAL_gdr_uebernahme.md:186`). #280, the
  driver-free workaround, is **closed as impossible in userspace**
  (`cudaHostRegisterIoMemory` -> `CUDA_ERROR_INVALID_VALUE`, driver guard
  `osCheckGpuBarsOverlapAddrRange`).

### 1.2 Volatility / reconstructibility classes

The fork already has this taxonomy — in prose, in one place, unenforced.
`dual_group_lane.py:5061-5063` (`LaneLending`, #274):

> the borrower may only put EVACUABLE content there — discardable scratch, or
> spillable session KV. Permanent posts are refused.

#407 promotes those three words to a checked property. It does **not** invent a
parallel taxonomy: the object classes are `offload_register.OFFLOAD_CLASSES`,
already eight strong, and the volatility class is an *attribute* of each.

| Class | Meaning | Members (with the site that fixes them there) |
|---|---|---|
| **RECONSTRUCTABLE** | can be dropped and regenerated at a known, bounded cost; no evacuation path needed | `graph_rungs` (recapture), `lane_workspaces` (scratch), `drafter_heads` and `experts` (re-readable from the checkpoint — this is what makes the #389 NVMe tier legal at all), `kv_shadow` (`offload_register.py:112-122`: *"the only class whose park/discard is FREE — there is nothing to copy back"*) |
| **EXPENSIVE** | reconstructable only by redoing user-visible work; must be evacuated, never dropped | session KV (#224: the whole point is *"no re-prefill, no output cap"*), `cold_lane`, `gdn_state_sets` **in its exported/SUSPENDED form** (#461) |
| **PERSISTENT** | correctness depends on surviving process exit, and in #89's case a reboot | hibernate images (#89), and by inheritance #306's compressed form of them |
| **DEVICE-BOUND** | may not leave local VRAM at all; not a preference, a named invariant | **LIVE** GDN/Mamba recurrent state — `kv_session_spill_destination.py:35`: *"It never travels, is never quantized, never crosses the wire … the most corruption-sensitive payload in the system (recurrent error accumulates)"*. The row above holds the same content once evacuated; see X2 and `DESIGN_286` §8 (#461) |

The fourth row is why admission has to be a *refusal* mechanism and not a
ranking. A registry that merely ordered tiers by cost would happily offer the
GDN pool a remote tier.

Two class properties fall out of the survey and belong in the schema from cut 1,
because retrofitting them is a schema change:

* **Persistence is currently assumed, never declared.** `--hibernate-dir`
  pointed at a tmpfs produces a "hibernate" that does not survive a reboot, and
  nothing in the tree would notice. This is the single clearest thing the
  registry buys (§2.4).
* **Access-frequency budget.** #306 is scoped as an idle-workbench tenant that
  rewrites cold blobs in place; it needs to know a tier is cold enough that CPU
  compression time is free. No current vocabulary expresses that.

### 1.3 A naming hazard to settle now

**"cold tier" is overloaded in this tree.** `ANALYSE_389` and `ANALYSE_393` use
it for *host-DRAM-resident MoE experts* with a 145 ms/token bound; #306 uses it
for *hibernate images on disk*. They are different tiers, different volatility
classes, and different consumers. This document uses the tier ids of §3.1 and
the word "cold" only with an explicit qualifier.

---

## 2. Consumer survey

Eight consumers, read in their target-selection code. Judgement is
**MECHANICAL** when the migration is swapping a constant or a list for a
registry query, and **REBUILD** when the consumer's control flow encodes
something the registry contradicts.

### 2.1 Expert offload #77/#123 (+ #125 / #254 / #390 / #394)

`layers/moe/expert_offload.py`, with call sites in
`layers/moe/fused_moe_triton/layer.py` and allocation-time forcing in
`layers/quantization/{fp8.py:1301, awq/schemes/awq_moe.py:78, gptq/schemes/gptq_moe.py:251}`.

**Today.** There is no tier variable. The target is the literal
`torch.empty_like(spill_src, device="cpu").pin_memory()`, written out five
times (`install` at `:1995-2038`, `stage_experts_into_tiers` at `:1055`,
`StreamingExpertStager._ensure_tiers` at `:1422`, `freeze_from_source` at
`:2682`, and `:2784`). Worse, the three quantization loaders pre-commit the
target before any cache object exists:

```python
# fp8.py:1301
_moe_dev = ("cpu" if envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.get() < 1.0 else None)
```

How much stays resident is not a capacity computation at all — it is a fraction
of *expert count* (`resident_slot_count`, `SGLANG_MOE_RESIDENT_EXPERT_FRACTION`,
default 1.0) plus `scratch = max(8, R // 4)`. Nothing reads free VRAM, nothing
reads `MemAvailable`.

**Identity is inconsistent inside one file.** `_pcie_link_gbps_by_uuid` (`:655`)
resolves through the #331 `IdentityMap` and says why in its own docstring
(*"#392 is what happens when those are conflated"*), while every device buffer
is allocated at `torch.cuda.current_device()`.

**The dead capability — and the highest-yield cut in this document.**
`derive_link_weights` / `resolve_host_shard_ratio` (#394) size each rank's share
of the cold expert pool by its H2D link instead of splitting equally, and the
arithmetic is already recorded: 0.93 GB over 6.4 GB/s = **145 ms/token** with
equal shards against 2.79 GB over the 32.4 GB/s the three links absorb together
= **86 ms/token**, a **1.69x** ceiling on the cold tier
(`expert_offload.py:509-523`, `ANALYSE_393_ik_llama.md` §7.3/§7.4, which rates
it *"highest yield/effort item in this document"*). It is merged, tested, and
**unreachable in production**: `_gguf_cold_shard_context` (`layer.py:1271-1295`)
calls `cold_shard_context(rank, world)` with no `card_uuids`, deliberately,
because gathering the UUID vector would need a collective inside the weight-load
loop. Its docstring: *"the ratio resolves from `SGLANG_MOE_HOST_SHARD_RATIO` or
not at all."*

**Migration cut.** Two halves with very different costs.
*Half A (MECHANICAL, S):* the registry publishes the rank -> card-UUID vector
from `IdentityMap` at startup, without a collective; `resolve_host_shard_ratio`
gets its argument. Nothing else moves.
*Half B (REBUILD):* a third tier under the pinned host pool. `MoEExpertOffloadCache`
holds exactly two dicts (`_resident`, `_pinned`) and `ExpertResidencyPlanner.resolve`
returns a binary `resident | spill` partition with `id >= resident_count` as the
tier test; the marlin `moe_align` tiling depends on the buffer being a fixed
`R + C`. Additionally, `device_view_of_pinned` (`:1721`) builds a
`__cuda_array_interface__` over the same virtual address — a property of pinned
host RAM and of nothing else — and the CUDA-graph-capturable path is built
entirely on it. A third tier is a new return shape for `resolve`, a new wave
planner, and a new fetch mechanism.

**Judgement: MECHANICAL for half A, REBUILD for half B.**

### 2.2 HiCache L1 / L2 / L3

`mem_cache/hiradix_cache.py`, `managers/cache_controller.py`,
`mem_cache/pool_host/*`, `mem_cache/storage/backend_factory.py`.

**Today.** The ladder is not data. L1 = device pool, L2 = `HostKVCache`
subclasses, L3 = a single attached `HiCacheStorage` backend. The only enumerable
tier list is `StorageMedium` (V3 above), which is emitted into KV-cache events
and never influences placement. Capacity exists for L2 only — `--hicache-ratio`
(default 2.0) or `--hicache-size` (GB), min-all-reduced across ranks by
`sync_fixed_hicache_size` so uneven-TP ranks agree. **L3 has no declared
capacity at all.** Bandwidth and latency exist only as *post-hoc per-backend
metrics* (`prefetch_bandwidth` / `backup_bandwidth` in the hf3fs and mooncake
backends), drained into telemetry, consumed by nothing.

**What breaks.** The ladder is the type signature of every controller method,
not a configuration: `write()` takes device indices and allocates host indices;
`load()` does the reverse; `write_storage()` and `prefetch()` both take **host**
indices, and the zero-copy path is literally
`batch_get_v1(hash_values, host_indices, ...)`. L3 is unreachable without
burning an L2 page. `TreeNode` carries exactly `value` and `host_value` — two
slots, not a list. `attach_storage_backend` refuses a second backend, so
"multiple L3 tiers" is not expressible.

**Migration cut.** Split it. The registry owns *which* L3 and *how big* L2 —
`StorageBackendFactory` is already a registry with lazy `importlib` loading,
dynamic registration and runtime attach/detach, so pointing its `choices=` list
and `_create_builtin_backend` if/elif chain at the tier registry is genuinely
mechanical, and feeding L2 a declared capacity is additive. The **ladder itself
is out of scope** (§5, exclusion X1).

**Judgement: MECHANICAL for backend selection and L2 capacity; the ladder is
excluded, not deferred.**

### 2.3 KV / session spill #224 ("kvso")

`managers/kv_session_offload.py` (which session) +
`managers/kv_session_spill_destination.py` (where).

**Today — and a correction to the brief.** The *victim* half is already
target-agnostic and needs no change at all: `select_spill_victim` over
`session_priority_key = (spill_class_rank, is_fast_lane, -arrival_seq)`, tabu on
the oldest normal session, classes `SPILL_CLASSES = (preferred, normal, never)`.
The *target* half has its own vocabulary (V2), and it is **not** the one that
gained peer-VRAM and remote entries — that is #286's V1, where `"remote"` is an
explicit stub whose named attachment point is #224
(`offload_movement.py:715`: *"remote: stub tier (#224 RDMA attachment point, not
wired yet)"*). So #286 has a hole shaped like #224, and #224 filled it with a
different vocabulary instead. That divergence is the single strongest argument
for this task.

Tier 0 is not in the destination list at all: it is the pinned host pool object,
sized by `--kv-session-offload-host-ram-gib`. Every further entry is a HiCache
backend name. Runtime choice is **positional, not capability-based**:
`_start_park()` always uses `tiers[0]`, and failover walks `tier_index + 1` on
I/O failure only — there is no capacity or pressure query to route on. Capability
is inferred from a name string (`pointer_io = name == "mooncake"`), the exact
anti-pattern a registry removes.

**What breaks — contradiction C1.** `destinations_error` rejects any list whose
first entry is not `local`, and gives a physical reason:

> the device D2H copy lands in locally pinned host RAM, so every further tier
> stages through the local one; and the measured cross-rig path (3.43 GB/s RDMA
> write, 1.47 us latency, PCIe-3.0-x4-bound) is orders of magnitude below local
> host RAM — a remote tier is a tier BELOW local, never above it.

Two claims are fused there. The **staging** claim is true for blob-store tiers
and is a *graph* fact. The **bandwidth** claim is a total order that peer-VRAM
falsifies: a D2D park does not stage through host at all. (In passing, the two
numbers in that message come from two different links — 3.43 GB/s is the
slot-limited 100G line, 1.47 us is the 40G one. Worth fixing when the site is
touched.)

**Migration cut.** `SUPPORTED_PARK_BACKENDS` and `ALL_STORAGE_BACKENDS` become
registry queries filtered by capability rather than by name — mechanical, and it
retires the sync-test that keeps the mirror honest. The `local`-first law becomes
a *reachability* constraint computed from the staging graph (§3.5), which is a
real change to `destinations_error`'s shape. `bundle_spillable_sizes` (`:111`)
is already declared as *"the factoring seam for the future GDN-state host tier"*
— that is where per-class admission plugs in, and it costs nothing to use.

**Judgement: MECHANICAL for the list; REBUILD for the ordering law. Victim
selection: no migration.**

### 2.4 Hibernate #89

`model_loader/hibernate.py`, flags at `server_args.py:5232-5249`, auto-detect at
`:12565-12602`.

**Today.** No target selection exists. One string flag, `--hibernate-dir`, and
every path is `os.path.join` under it. Per-rank shards are `rank{tp}_{GPU-uuid}.pt`
and cross-rank serialisation is `fcntl.flock` on `.park_lock_{uuid}`.

**What breaks.** Three things, one of them a correctness issue.
(a) POSIX semantics are load-bearing — `os.makedirs`, `open()`, and above all
`flock`. An object-store or remote tier has no flock, so only filesystem-class
tiers are admissible and that must be a *capability filter*, not a name check.
(b) **Persistence is assumed and never declared** (§1.2). This is the silent
regression a naive migration would introduce and the reason the volatility class
must exist in cut 1's schema.
(c) The query must be answerable **at argument-parse time, without CUDA**. The
existing code already dodges exactly this trap — the coarse gate uses
`list_devices`, not `identity_map`, because *"the CUDA half of the map would
create a context there"* (a few hundred MiB per visible card).

**Judgement: MECHANICAL-PLUS.** One-line path swap, but only after the registry
carries a persistence class and a driver-free enumeration mode.

### 2.5 Residency ladder #305

`docs/dev/DESIGN_305_multi_model_serving.md`, `registry/rungs.py`,
`registry/ledger.py`, `registry/arbiter.py`.

**Today.** The cleanest situation of the eight: the ladder has *no mechanism at
all*. `rungs.py` states it in capitals — *"NOTHING HERE MOVES A MODEL. Rungs are
declared and reported; transitions are the later cuts."* The seam is
`transition_refusal(target_rung)` (`:253`), a function that returns an error
string naming the cut that would implement the move. `rung_of()` maps
`TenantState` onto `Rung`; `promote_cost_of()` returns an *a-priori* cost class
with `measured: bool = False`, kept in a different field from the registry's
`measured_promotion_ms` on purpose, so a class can never be read as a
measurement.

**What breaks — and what must not.** `TenantState` is
`{HOT, WARM_GPU, WARM_HOST, COLD}` and it is **client-visible**: `/v1/models`
reports it verbatim through `entrypoints/openai/registry_view.py`, which
documents the honesty rule ("not as a boolean available"). Two of its four
members are tier *names* inside a policy state enum. With five tiers this enum
cannot be widened without changing a published API. The design keeps
`TenantState` frozen and adds an **orthogonal** `resident_tier: TierId` field:
`WARM_HOST` continues to mean "warm, not holding device bytes on its own card",
and the exact tier is named beside it. `GPU_RESIDENT_STATES` — the frozenset
that gates the ledger invariant — stays exactly as is.

That frozenset also exposes the peer-VRAM consequence: a tenant that parked into
*another card's* VRAM is not `WARM_HOST` and does still hold device bytes, on a
card it does not own. `registry/ledger.py` structurally supports this (one file
per card, holder identity separate from the card) but nothing has ever written
such a reservation.

**Judgement: NO MIGRATION.** #407 is the mechanism #305 declared and did not
build. The risk here is inverted: not migration cost, but shipping a second
ledger and violating §6.

### 2.6 Short-term offload register #286

`model_executor/offload_register.py` (policy),
`model_executor/offload_movement.py` (mechanism + `CapacityLedger`),
`model_executor/offload_bus_budget.py` (PCIe arbiter),
flags at `server_args.py:4583-4620`.

**Today — the best-shaped consumer, and the right donor.** `_order_for()` and
`_select_target()` (`offload_movement.py:687,706`) already do what a registry
should answer, and they do it by *capability*, not by name: `remote` is always
skipped as a stub; `TagPayload`/`SuspendPayload` park to `host_ram` only;
`va_stable_required` items refuse plain tensor copies; `peer_vram` requires both
a **probed directed P2P path** (`probe.path_admits`) and an explicitly granted
budget, and degrades to `host_ram` with a log line otherwise; `host_ram` checks
`_has_headroom()`; exhaustion raises a `MovementError` naming every rung's
refusal individually. Lifetime is `TIME_CONSTANT_TIERS = ("wave", "phase", "turn")`
plus admission and pressure boundaries for two special classes. A parked item is
a **latency term** (`retrieval_latency_ms`), never a non-availability.

`offload_bus_budget.py` is the socket the registry's bandwidth number has been
waiting for: a consumer-neutral PCIe arbiter over named consumers
(`expert_streaming`, `stage2_phase`, `kv_spill`), weighted debt-model buckets,
and an **injectable total rate** (`set_measured_rate`; rate 0 = open budget =
byte-identical to today).

**What breaks.**
(a) **C2, identity.** `CapacityLedger` keys on `(target, cuda_ordinal)`;
`_ItemMovement.peer_device` and `PeerProbe`'s `(int, int)` keys are the same
ordinal. Legitimate while in-process; a #331 BUG the moment it is published.
(b) The universal-fallback rule is hardcoded:
`if "peer_vram" in order and "host_ram" not in order: order = order + ("host_ram",)`
(`:702`). A registry with no host_ram tier dead-ends.
(c) `PARK_TARGETS` lives in `offload_register.py` rather than the movement half
**deliberately**, *"so the policy parser can name them without importing the
movement half"* — the registry import must respect the same layering.
(d) `set_budget("peer_vram", dev, nbytes)` carries a standing TODO for the
dual-group budget feed, so peer-VRAM is currently unreachable in practice: no
budgeted device means *"no peer device with a granted budget"*.

**Migration cut.** Lift `_select_target` and `CapacityLedger` into #407 as the
reference implementation rather than writing new ones, widen the key from
`(target, ordinal)` to a `TierId`, and back the peer-VRAM budget with
`registry/ledger.py` so (d) is answered by the ledger that already owns the
card. `PARK_TARGETS` becomes a class filter over registry tiers so the shipped
CLI strings keep working.

**Judgement: MECHANICAL.**

### 2.7 NVMe expert tier #389 (unbuilt)

`docs/dev/ANALYSE_389_nvme_expert_tier.md`.

Nothing to migrate; treat it as the **requirements document**, because it is
already written in registry vocabulary. It assumes a strictly additive third
rung — *"the NVMe tier below #77/#123"* — under an unchanged host-RAM tier, and
its placement rule is a single size threshold (`<= ~150 GB of 4-bit weights fits
the existing tier`), not a cost model. Its verdict for K3 on this box is
0.106 tok/s against #77's measured 6.97 tok/s on the existing RAM tier, and it
is disciplined about *whose* number that is: *"the 1.8 GB/s is this box … this
verdict must not be recorded as a verdict on the approach."* That sentence is
the rig-is-lower-bound rule, and it belongs in the registry as a provenance
field, not a comment.

Four requirements it imposes on the schema: per-tier capacity that can say "this
tier does not fit the model" (729 GB free vs a 982 GiB container); per-tier
bandwidth as a first-class measured attribute; a distinction between the raw
device and a cache in front of it (the ARC rows are excluded from the tier's
bandwidth on purpose); and per-tier hit-rate telemetry — cut A shipped as
`layers/moe/expert_stats.py` but **the number is still empty**, with a known
blind spot (captured decode under `SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` is
uncounted).

**Judgement: GREENFIELD — the registry's best-shaped customer.** Note also that
#389's NVMe and HiCache's L3 can be the same physical device, which is an
argument for one registry rather than two.

### 2.8 Cold-tier compression #306 (unbuilt)

Two roadmap bullets, identical wording:
`ANALYSE_347_cross_feature_optimizations.md:29` and
`DESIGN_347_idle_workbench.md:215-216` (*"None is built here"*).

It inherits every one of #89's assumptions transitively, and adds two schema
requirements nothing else needs: an **access-frequency / reanimation-latency
budget** (it is scoped as a preemptible idle-workbench tenant, so it needs to
know the tier is cold enough that CPU time is free) and the fact that a tier's
contents may be **rewritten in place in a transformed representation** by a
background tenant the producer never hears about.

**Judgement: GREENFIELD.** Its only claim on cut 1 is a capability flag; design
for it now or it forces a schema change later.

### 2.9 Summary

| Consumer | Target selection today | Migration cut | Verdict |
|---|---|---|---|
| #77/#123 expert offload | `device="cpu"` + `.pin_memory()` x5; residency is a fraction of expert *count* | A: publish rank -> UUID vector, revive #394. B: a third rung under the pinned pool | **MECHANICAL** (A) / **REBUILD** (B) |
| HiCache L1/L2/L3 | closed `choices=` list; single attached L3; L2 sized by ratio | registry owns which L3 + L2 capacity; ladder excluded | **MECHANICAL** (backends) / **excluded** (ladder) |
| #224 kvso | `("file","mooncake","dynamic")`, `local`-first law, positional `tiers[0]`, error-only failover | list -> capability query; `local`-first -> staging-graph reachability | **MECHANICAL** (list) / **REBUILD** (ordering law); victim half unchanged |
| #89 hibernate | one flag, POSIX + flock + NVML-UUID shard names | path from a `persistent=True`, filesystem-class tier | **MECHANICAL-PLUS** (needs persistence class + driver-free query) |
| #305 residency ladder | none — `transition_refusal()` names the missing cut | consume the registry as its mechanism | **NO MIGRATION** (constraint: no second ledger) |
| #286 offload register | `PARK_TARGETS` + capability-driven `_select_target` + `CapacityLedger` | lift both into #407; widen key to `TierId` | **MECHANICAL** (and the donor) |
| #389 NVMe expert tier | unbuilt | — | **GREENFIELD** (requirements source) |
| #306 cold-tier compression | unbuilt | — | **GREENFIELD** (capability flag only) |

Two further fragmentations the survey turned up, both cheap to absorb and
expensive to leave: host-RAM capacity is queried ad hoc through
`psutil.virtual_memory()` in eight unrelated sites, and free disk space through
`os.statvfs` in three more (`nixl_cleaner`, `lru_file_evictor`, and the #341
training feasibility gate). Neither has an owner.

---

## 3. Registry interface

Module: `python/sglang/srt/memtier/`. **Not** `srt/registry/` (that name is
taken by the engine registry, #333/#305) and **not** `mem_cache/registry.py`
(taken by the RadixCache backend factory). Stdlib-only at module scope; NVML,
torch and `uneven_perf` imported lazily inside the functions that need them, per
`cost_model.py`'s precedent (4 ms on top of a warm `sglang` import).

### 3.1 Tier descriptor

```
TierId          str, structured, never positional. Grammar:
                  "vram:GPU-<uuid>"          local or peer device memory
                  "host:<hostname>"          host RAM of one machine
                  "fs:<hostname>:<mount>"    a filesystem (NVMe, tmpfs, ...)
                  "blob:<backend>:<scope>"   a HiCache-class blob store
                The card half is the #331 identity: UUID primary, PCI BDF as
                the readable secondary. A CUDA ordinal never appears in a
                TierId; it is resolved live, in one call, at the moment an API
                needs it (registry/planner_bridge.py:59 is the precedent).

TierDescriptor  id: TierId
                kind: TierKind{DEVICE, HOST, FILESYSTEM, BLOB}
                host: str                  # the machine it is attached to
                capacity: TierCapacity
                volatility: Volatility{RECONSTRUCTABLE_OK, EXPENSIVE_OK,
                                       PERSISTENT, DEVICE_BOUND_ONLY}
                admits: frozenset[str]     # OFFLOAD_CLASSES it may hold
                caps: TierCaps
                health: TierHealth
                properties: Mapping[str, str]   # flock_available,
                                                # pointer_io, rewritable,
                                                # shared_fingerprint, ...
```

`properties` is where the survey's name-derived capability checks land:
`pointer_io = name == "mooncake"` becomes `props["pointer_io"]`, and #89's flock
dependency becomes `props["flock"]` instead of a filesystem-kind assumption.

### 3.2 Capacity is not one number

`DESIGN_330_vram_dial.md` §2 already separates a rank's VRAM into a **pinned
floor** (weights, graphs, GDN pool, activation workspaces, allocator metadata —
measured once as `nvml_process_used_bytes - vmm_backed_bytes`) and a **dialable
span** (the VMM-backed KV pool tail). A single "free bytes" field would erase
that distinction and #330 would have to re-derive it.

```
TierCapacity    total: Rate            # measured, or a named absence
                floor: Rate            # not reclaimable at this tier
                reserved: int          # from the ledger, cross-process
                corridor: int          # #330's 400 MiB, DEVICE kind only
                headroom() -> Rate     # total - floor - reserved - corridor
```

For `DEVICE` tiers every one of those fields is **read from
`registry/ledger.py`**, not recomputed: that module states *"There is exactly
one implementation of the invariant on this rig, and this is it."* #407 extends
its file shape (one JSON per tier id under `/run/htsglang/`) to non-device kinds,
where `corridor` is zero and the invariant is plain capacity. It does not
re-implement leases, heartbeats, reaping or the exclusive window.

`waste(T) = reserved - measured` carries over unchanged, including the rule that
it is **reported and never acted on**.

### 3.3 Measured caps, and the byte-cost hook

```
TierCaps        latency_class: Rate    # us, MEASURED | ESTIMATE | ABSENT
                bandwidth: Rate        # GB/s, ditto
                aperture_bytes: Optional[int]   # effective, never nominal
                ledger_key: str        # the accounting bucket in #400 terms
```

Every field is a `cost_model.Rate`, reused verbatim — one number with its
provenance, `value is None` exactly when `provenance is ABSENT`, `require()`
raising `AbsentRate` with the absence's own text. #348b's D4 fixed precisely the
defect this inherits: a missing `membw` used to read downstream as an extremely
slow but *valid* card. A missing tier bandwidth must not read as an extremely
slow but usable tier.

The byte-cost accounting hook is one line and it already has a socket:
`bus_budget.set_measured_rate(consumer, tier.caps.bandwidth.require(...))`.
`offload_bus_budget.py` is consumer-neutral, its default rate 0 is the open
budget (byte-identical to today), and its three named consumers are exactly
expert streaming, stage-2 phase moves and KV spill.

`aperture_bytes` is **effective, never nominal**, following #286's own rule
(*"no aperture constant lives in this code"*) and
`scripts/p2p_readiness/verdict_diff.md` V6. On this rig the nominal 256 MiB BAR1
maps 96 MiB contiguously, and #278 W1 showed BAR size moves nothing at any
measured step — so a nominal number would be both wrong and misleading.

### 3.4 Health and reachability

A remote tier can vanish, and it must vanish *by name*. The tree already has the
surface: `planner/rig_coupling.py:418` `GateRow` — `key, label, verdict, reason,
remedy, local, remote, provenance, evidence, register_key`, verdicts
`ok`/`warn`/`block`, provenance `measured`/`estimate`/`absent`, sides
`dashboard`/`pve-host`/`far-rig`. `EVAL_gdr_uebernahme.md:530` records the cost
of adding one: *"one `_row_*` builder plus one append — no schema change."*

The registry therefore emits `GateRow`s rather than defining a health model, and
`TierHealth` is thin:

```
TierHealth      reachable: bool
                verdict: str           # ok | warn | block, GateRow's vocabulary
                reason: str            # never empty when not ok
                last_seen_s: float
```

Three degradation rules, all named rather than silent:

* An unreachable tier is **enumerated with `verdict="block"`**, never omitted.
  Omission is how a spill target silently becomes a different spill target.
* An **unmeasured** path is refused for admission, not assumed usable — #286's
  existing rule (*"an unmeasured path is NEVER assumed usable for parking,
  however tempting"*), lifted unchanged.
* A remote tier carries a **producer fingerprint** as a property. #224 already
  requires that a remote entry is a hit only if producer == consumer on model,
  quantization, KV dtype, TP size / rank-tp-ratio, DCP geometry, head geometry
  and layer count; a mismatch is a hard request error, never silent corruption.
  The registry surfaces the fingerprint so the check is not re-implemented per
  consumer.

### 3.5 The query API, and how C1 is resolved

```
targets_for(object_class: str,
            bytes_needed: int,
            *,
            origin: TierId,
            require: Mapping[str, str] = {}) -> TargetList
```

`TargetList` is an **ordered candidate list whose ordering key is exposed**, plus
a `refusals: Tuple[Refusal, ...]` naming every tier that was considered and
rejected, with the reason. That shape is copied from `_select_target`'s
`MovementError`, which already names every rung's refusal individually — a good
property, and one that makes a policy layer able to disagree.

Ordering is **not** a constant ladder. It is a shortest-path over a **staging
graph** whose nodes are tiers and whose edges are the moves that physically
exist:

* `vram:X -> host:H` — the `h2d`/`d2h` rates from `card_probe` (E4);
* `vram:X -> vram:Y` — only if `PeerProbe.path_admits` and the pair has a
  measured `Hop` (E1) or `PathProfile` (E2);
* `host:H -> fs:H:*` and `host:H -> blob:*` — the storage path;
* `host:H1 -> host:H2` — the wire.

This is what resolves **C1**. #224's rule that `local` must come first is
recovered as a *reachability* fact — for a `blob:` target the only edge out of a
device tier runs through `host:`, so `local` is on every path by construction —
while `peer_vram` above `host_ram` is recovered because `vram -> vram` is a
direct edge that does not pass through `host`. Neither consumer's shipped
behaviour changes; the law stops being a hardcoded total order and becomes a
consequence of the graph. `destinations_error` keeps rejecting a list that omits
`local` before a blob tier, and now says so with the edge as its reason.

**Ordering key**, exposed on every candidate so a policy layer can substitute
its own: `(volatility_admissible, headroom_sufficient, path_cost_ms(bytes),
provenance_rank)`. `path_cost_ms` composes the graph edges through
`cost_model.allreduce_seconds`' sibling for point moves; `provenance_rank`
demotes an `ESTIMATE` below a `MEASURED` of similar cost, so the registry never
prefers a guess to a measurement.

### 3.6 What the registry does not do

* It does not pick. `targets_for` returns candidates; #305's arbiter, #286's
  policy half and #363's controller pick.
* It does not move bytes. `offload_movement.RealMovementBackend` moves bytes.
* It does not decide **where compute happens**. That is #302's axis (§6).
* It does not own the L1/L2/L3 ladder (§5, X1).
* It does not price a collective. `cost_model` prices collectives; the registry
  prices a *point move to a resting place*, and delegates card-pair edges to
  `PairMatrix` rather than storing a second copy. (`cost_model`'s rule that "a
  hop is between two different cards" survives: no tier edge is card-to-itself,
  and `vram:X -> host:H` has no card on its far end.)

---

## 4. Measurement plan

The registry ships every number with its provenance, so a missing measurement is
a visible `ABSENT`, not a blocker. These are the ones worth taking, ordered by
what they unblock.

| # | Missing | Who produces it | Cost | Unblocks |
|---|---|---|---|---|
| M1 | **BAR1 point-latency ladder, 8 B - 64 KiB** | Promised as W1 in `EVAL_gdr_uebernahme.md:656` (*"extend the ladder to 16/32/256 KiB to bracket 20 KiB and 80 KiB"*), ~30 min, no server boot. **Structural caveat: barlink implements collectives only and has no send/recv** (`scripts/probe/barlink_vs_nccl.py:4-22`), so this cannot come from the barlink harness — it needs the p2p_readiness `d2d_bench` path or the GDR bench with a local target | S | T1's latency class, which is today an *assumption*. Rides the comm-suite as a new arm |
| M2 | **Effective BAR1 aperture per directed pair** | `scripts/p2p_readiness/capability_matrix.py` — the package exists and **has never been run**; no `results/` directory | S | `TierCaps.aperture_bytes`; #286's window policy (`reject` vs `chunk`) is currently defaulted conservatively for want of this |
| M3 | **Direct D2D bandwidth vs host staging, per direction and size** | `scripts/p2p_readiness/d2d_bench.py`, same run as M2 | S | The `vram -> vram` edge weight in §3.5's graph; today only E2 has a loader and no file to load |
| M4 | **Host DRAM read bandwidth** | Any sustained-read probe; `ANALYSE_393_ik_llama.md:554` names it as the open item (*"a measured DRAM read bandwidth at the top of the 32-45 GB/s band"*). Note all three rank processes contend for one controller | S | T2 goes MEASURED. It is load-bearing: 73 ms/token (2.79 GB / 38 GB/s) is the floor for any RAM-resident expert tier, and that floor is currently an estimate |
| M5 | **NVMe read latency** (any unit) | No `fio`, no `iodepth`, nothing in the tree. #389's arithmetic is bandwidth-only | S | T3's latency class; #389's placement threshold is a size rule today and would stay one without it |
| M6 | **ZFS write-back behaviour for spill-to-disk** | Extend the `a8a2f7bc22` harness. That commit measured the *read* side conclusively — `posix_fadvise(DONTNEED)` is a no-op for mapped pages **and on the ZFS pool even post-unmap** (16384/16384 pages resident by `mincore`), while `madvise(MADV_PAGEOUT)` reclaims to 0. The write side is unmeasured | S-M | Whether a `fs:` tier's writes are durable when the call returns, and whether the page cache they leave behind is reclaimable in time. Boot 8 died with `memory.current` at 98.3/98.5 GiB for seven minutes: *reclaimable is not reclaimable in time* |
| M7 | **Rig-2 point reads**, and cross-rig GPU-to-GPU | `EVAL_gdr_uebernahme.md:186` (§8.2): GPU-to-GPU across the rig boundary was **never demonstrated** — *"exactly the NORDSTERN shape"*. #280's userspace workaround is closed as impossible | M | T4's VRAM half and T5 entirely. Not on the #407 critical path; the registry declares them `ABSENT` and remains correct |
| M8 | **MR registration cost** (`cuMemCreate` -> export -> two ioctls -> `ibv_reg_dmabuf_mr`) | Untimed, `EVAL_gdr_uebernahme.md:174-176`: *"for a static pool amortised to zero; for anything dynamic it is the whole design"* | S | Whether a remote tier can be allocated per-session or must be a static pool |

**Artifacts that already carry numbers**, and what each still lacks:

* **comm-suite** (`planner/comm_suite.py`, #271) — ten arms, <=90 s budget, an
  A-vs-A noise floor arm that runs *first*, and four honest states per arm where
  `absent` carries a one-sentence reason so "nobody measured this" and "this is
  zero" stay distinguishable. It feeds `planner/rig_artifact.py`, which brings
  curation, an anonymisation gate, preview and submit — a second source must not
  be a second copy of the scrub. **It has no BAR1 arm, no storage arm and no
  host-DRAM arm.** M1, M4 and M5 are three new arms in an existing harness,
  which is why they are S.
* **#278 GDR matrix** (`INTEGRATION_R3_VALIDATION.md:6176-6210`) — the clean
  retake on an exclusive box, honest about a **bimodal 80 KiB point** (16 vs
  49 us in identical runs) and reporting only effects >1.5x. It overturned two
  earlier verdicts (no crossover *point*, a band at 64-80 KiB; BAR size
  irrelevant) and confirmed V5, the NIC serialisation wall. Open in its own
  words: the symmetric-load p50/p99 asymmetry (deferred to #279), an
  unexplained `WC remote invalid request error` on one directed pair, and no
  cross-boundary GPU-to-GPU.
* **ANALYSE_389** — the NVMe bandwidth ladder with ARC rows explicitly excluded
  from the tier's number, plus free-space and container-size arithmetic. Carries
  no latency.
* **card_probe** (`rigmon/card_probe.py`) — the per-card `membw` / GEMM / fp8 /
  H2D / D2H source and the ordered pair matrix, 64 MiB pinned transfers,
  best-of wall clock, cached under a path keyed on **sorted UUIDs**.

---

## 5. Cut plan

No big bang. Each cut is independently shippable, default-off or read-only, and
named by what it retires.

| Cut | Content | Effort | Yield |
|---|---|---|---|
| **1** | `srt/memtier/` module: `TierId`, `TierDescriptor`, read-only enumeration of T0-T3 from `IdentityMap` + `registry/ledger.py` + `psutil` + `statvfs`, provenance on every number, `GET /registry/tiers` beside the existing `/registry/cards`, one `GateRow` per tier. **No consumer changes. No writes.** | **S** | The enumeration becomes inspectable and the `ABSENT` fields become visible as gaps rather than as defaults. Retires nothing yet; blocks nothing |
| **2** | Publish the rank -> card-UUID vector at startup from `IdentityMap`, **without a collective**; pass it to `resolve_host_shard_ratio`. #394's merged, tested link-proportional sharding comes alive | **S** | **The largest quantified yield in this plan: 145 -> 86 ms/token on the cold tier, a 1.69x bound** (`ANALYSE_393` §7.3/§7.4). Retires the `SGLANG_MOE_HOST_SHARD_RATIO` hand-typed vector as the only route to a non-equal split |
| **3** | #224 consumes the registry: `SUPPORTED_PARK_BACKENDS` / `ALL_STORAGE_BACKENDS` become capability queries; `destinations_error`'s `local`-first law becomes staging-graph reachability; per-class admission hooks into the existing `bundle_spillable_sizes` seam | **S-M** | Retires V2 and the mirror sync-test. Resolves **C1**. Makes GDN's device-bound invariant machine-checked instead of prose |
| **4** | #286's `CapacityLedger` becomes a view of `registry/ledger.py`; keys widen from `(target, ordinal)` to `TierId`; the peer-VRAM budget TODO is answered by the ledger that owns the card; `PARK_TARGETS` becomes a class filter so the shipped CLI strings keep working | **M** | Resolves **C2**. Satisfies #305 §6 (one ledger). Makes `peer_vram` reachable in practice for the first time. Retires V1 as a vocabulary while keeping it as a CLI alias |
| **5** | Expert offload host-tier selection through the registry: the five `device="cpu"` literals and the three `_moe_dev` sites resolve a tier. Requires the registry to answer at weight-creation time, i.e. very early in load | **M** | Retires the last consumer with no tier vocabulary at all. Couples with cut 2's link ratios: same UUID vector, one source |
| **6** | #89: `--hibernate-dir` resolves through a `persistent=True`, `flock`-capable filesystem tier; registry gains a driver-free enumeration mode for the arg-parse-time gate | **S** | Closes the silent-correctness hole where a tmpfs `--hibernate-dir` produces a hibernate that does not survive a reboot |
| **7** | #305 transitions consume the registry as their mechanism; `resident_tier: TierId` added beside the frozen `TenantState` | **M** | `transition_refusal()` starts returning fewer refusals. Blocked on cut 4 |
| **8** | HiCache: `StorageBackendFactory`'s `choices=` list and `_create_builtin_backend` chain become registry lookups; L2 gets a declared capacity. **Ladder untouched** (X1) | **S-M** | Retires V4 and V3-as-a-second-enumeration |
| **9** | Declare the NVMe tier (#389) and the compression capability flag (#306) | **S** for the declaration | Blocked on M5/M6, and on #389's own cut D. The registry makes both a data change rather than a schema change |

Recommended order is 1, 2, 3, 4, then 5-9 as capacity allows. **This departs
from the brief's ordering in one place, deliberately:** the brief put #224 at
cut 2 and expert-offload host selection at cut 3. Cut 2 above is the *first half*
of that expert-offload cut, it is S rather than M, it is the only cut in the
plan whose yield is already measured, and it does not depend on cut 1 landing
first (only on the UUID vector, which cut 1 produces as a side effect). #224
follows immediately at cut 3, unchanged in content.

### Exclusions, with hard reasons

Per the alles-mit-allem-kombinierbar rule, an exclusion needs a physical or
logical wall, named.

* **X1 — the HiCache L1/L2/L3 ladder itself.** Not deferred, excluded. The
  ladder is not configuration: it is the type signature of every
  `HiCacheController` method (`write_storage` and `prefetch` both take **host**
  indices; the zero-copy path is `batch_get_v1(hash_values, host_indices, ...)`),
  it is `TreeNode`'s two fixed slots `value` and `host_value`, and it is the
  contiguous-prefix backup invariant in `write_backup`. Reordering or bypassing
  a rung is a rewrite of three modules for a goal #407 does not have. The
  registry owns *which* L3 and *how big* L2, which is where the value is.
* **X2 — LIVE GDN/Mamba state never gets a tier.** Not a scope decision:
  recurrent error accumulates, so a lossy or reordered round trip is a
  correctness failure, and #224 already names it *"the most
  corruption-sensitive payload in the system"*. The registry expresses this as
  `DEVICE_BOUND_ONLY` admission, which is a refusal mechanism rather than an
  omission — the class is enumerated and every tier refuses it by name.

  **Scope correction (#461, 2026-08-03):** this classifies the content in a
  STATE, not the feature. Live state is device-bound. The same session's state
  once EXPORTED (`MambaPool.export_state_blob` → CPU tensors keyed by field
  name, restorable into any free slot) is a self-describing byte payload that
  #364's slot executor already puts on a #224 `DestinationTier`, and it asks
  the registry as `EXPENSIVE_RECONSTRUCTABLE` when it does. That is not an
  exception to the law — nothing device-bound travels; an evacuation produces a
  DIFFERENT payload, which is why the correct park of a GDN set is always
  vacate-then-move. #224's own "never travels" sentence is scoped to the KV-tail
  park path it describes, as `mem_cache/gdn_slot_executor.py:46-50` already
  states. Consumers name the state through
  `short_term_offload_register.ContentState` (default `LIVE`, the conservative
  answer); details and the code evidence are in
  `DESIGN_286_short_term_register.md` §8.
* **X3 — cross-rig GPU-to-GPU as a tier edge.** Physically undemonstrated on
  this hardware, and the userspace route is closed by a driver guard (#280,
  `osCheckGpuBarsOverlapAddrRange`). The registry declares `T4-VRAM` with
  `health.verdict = "block"` and the reason attached, rather than omitting it —
  so a rig where it works needs no code change.
* **X4 — the registry does not enumerate compute placement.** #302's axis (§6).

Explicitly **not** excluded, against expectation: multi-rank-per-card, uneven TP,
and dual-group runtime all compose with this design, because tier identity is
per-card and per-host rather than per-rank. Two ranks on one card see one
`vram:` tier and contend for it through the same ledger, which is the correct
model.

---

## 6. Cross-references

**NORDSTERN (TP=5 cross-rig).** The registry must not assume a single rig, and
the schema does not: `TierId` carries a host component from cut 1, and `host:`
and `fs:` tiers are per-machine by construction. Two NORDSTERN facts are
registry *data*, not registry *assumptions*: the NIC relay serialises (three
parallel pairs cost 2.34x latency for 1.28x aggregate — a `properties` entry,
not a bandwidth number), and rig-2's cards have no demonstrated direct GPU path
(X3). A registry that omitted unreachable remote tiers would make NORDSTERN
invisible until it worked; one that enumerates them with a blocking verdict
makes the gap a dashboard row.

**#302 expert placement — a separate axis, and the separation is load-bearing.**
Placement decides **where compute happens**; the registry decides **where bytes
rest**. They meet at exactly one point: a resting place must be reachable from
the card that will compute, which is §3.5's graph. `DESIGN_348b_cost_model.md`
§3.3 already documented #302's entry point as *"exactly the two axes and nothing
else"* — `ComputeRates.for_family(GEMM_FAMILY_MOE)` and `PairMatrix.hop` — and
added the rule that if #302 reaches into `profile["gpus"]` or `probe["pairs"]`
directly, the library is missing a primitive. #407 inherits that rule verbatim
for tiers: a consumer that reads `psutil.virtual_memory()` or `os.statvfs`
directly after cut 1 means the registry is missing a primitive.

**#348b cost model.** The registry is the node layer over #348b's edge layer,
and reuses `Provenance`, `Rate`, `AbsentRate` and `Hop` rather than defining
parallel types. `reconcile_pair_matrices(*matrices, tolerance=0.10)` already
exists as the primitive for cross-checking two sources of the same physical
fact; E1 vs E2 vs E3 (§0) is its next customer, and reconciling them is a cut-4
side effect rather than a separate task.

**#400 / #333 ledgers.** Three accounting conventions carry over unchanged
because each was learned the expensive way: the ledger is a **floor**, with
`unpriced` naming what cannot be known from config and `unbounded` meaning the
caller must **refuse rather than guess** (#400, after #349 arm L was accepted
and then died at 31.14 GiB in use on a 31.34 GiB card); a refusal **names the
card by uuid/BDF with an itemised table**; and `waste = reserved - measured` is
reported and never acted on. #400's known blind spot carries over too — a second
CUDA context in the TokenizerManager frontend process is invisible to every rank
ledger — and a tier registry does not close it.

**K3 fixposten (`ANALYSE_389` §3).** The rig-2 tier is where the K3-class
feasibility arithmetic would have to live if it is ever retried: 982 GiB
container against 729 GB free locally is a *capacity* refusal the registry can
express, which is the difference between "we tried and it OOMed" and "the plan
does not fit, named before a byte moved". #389's own verdict stands unchanged —
0.106 tok/s against #77's measured 6.97 on the existing RAM tier — and its
rig-is-lower-bound caveat becomes a provenance field rather than a paragraph
somebody has to remember.

---

## 7. Open items

1. **M1-M6 (§4).** Six measurements, five of them S, three of them new arms in
   an existing harness. None blocks cut 1.
2. **The three-way reconciliation of E1/E2/E3** (§0). Deferred to cut 4, where
   the identity widening makes the keyspaces comparable in the first place. Doing
   it earlier would mean reconciling a UUID-keyed matrix against an
   ordinal-keyed one, which is how #392 happened.
3. **`expert_stats.py` is wired and empty.** #389's cut A shipped the
   per-tier hit-rate instrument and no card time was ever spent on it; captured
   decode under `SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` is a known uncounted blind
   spot. Without hit rates the registry can price a move but not predict how
   often one happens.
4. **The 3.43 GB/s / 1.47 us pairing in `destinations_error`'s message** mixes
   the 100G and 40G links. Cosmetic, but it is a user-facing error string; fix
   it when cut 3 touches the function.
5. **`safetensors._drop_file_cache_after_load` is fadvise-only** — the same ZFS
   no-op class as `a8a2f7bc22`, flagged in the merge commit, own ticket. Any
   `fs:` tier that manages its own page cache on `/spinning` inherits it.

---

## 8. Cut 1 as built

`python/sglang/srt/memtier/` — `tiers.py` (identity, record, volatility law),
`profile.py` (measured numbers as data), `registry.py` (enumerate / filter /
refuse), `reservations.py` (named ledger posts), `probe.py` (measurement
catalogue), `profiles/rig1.json` (this rig, and only this rig). 82 hermetic
tests in `test/registered/unit/memtier/`, `base-a-test-cpu`. No consumer reads
any of it; nothing writes outside a test's temp directory.

**Followed as designed:** the module name `memtier` and its reasons; the
`TierId` grammar; `TierKind{DEVICE, HOST, FILESYSTEM, BLOB}` with locality as
a host field rather than a kind; `Volatility` with `DEVICE_BOUND_ONLY`;
`TierCapacity`'s total/floor/reserved/corridor split; `cost_model.Rate` and
`Provenance` reused verbatim; `TierHealth` in `GateRow`'s vocabulary, with
unreachable tiers enumerated and blocked rather than omitted; `admits` checked
against `offload_register.OFFLOAD_CLASSES`; the candidate-list-plus-named-
refusals shape of `_select_target`; the three honesty corrections shipped as
ESTIMATE (host DRAM) and ABSENT (peer-VRAM latency class, NVMe latency).

**Deviations, each with its reason:**

1. **`aperture_bytes` is a `Rate`, not `Optional[int]`.** An int cannot say why
   it is missing, and this field's absence (M2, never run) is the thing worth
   saying. Same rule as every other number in the record.
2. **`Volatility` (tier) and `PayloadClass` (content) are two enums.** §3.1
   carries one; admission needs both sides, and a single enum would have made
   the falsifier — a `persistence_required` payload offered a tmpfs tier —
   inexpressible. `ADMITTED_PAYLOADS` maps one onto the other in one table.
3. **`vram:unenumerated@<host>` added to the grammar.** X3 asks for rig-2's
   cards to be *declared* with a blocking verdict, and §3.1's grammar has no
   spelling for a card no local NVML has ever resolved. Inventing a UUID for
   it would have been worse. `is_bound` is false for these and
   `admission_refusal` refuses them whatever their volatility says.
4. **No staging graph, so no C1 resolution yet.** §3.5's shortest path is cut
   3's, together with the `destinations_error` it fixes. The ordering key is
   `(provenance_rank, -bandwidth, tier_id)` and is public on every candidate
   so the substitution is visible when it happens. `transport.stages_through`
   is recorded from cut 1 so the graph's edges are already data.
5. **No `GET /registry/tiers` route.** The brief for this cut is the registry
   plus records; an endpoint is a consumer. `TierRegistry.to_json()` and
   `gate_rows()` are the payload, ready to mount.
6. **Device tiers come from card-MODEL templates, not from the profile's tier
   list.** A membw figure is a property of a 5090, not of one particular 5090,
   and a profile that named UUIDs would be wrong on the next machine. Binding
   happens when a live card is enumerated; an unprofiled model gets live
   capacity and ABSENT caps, never a roofline.
7. **`_host_memory_bytes` reads `/proc/meminfo` rather than `psutil`.** §2.9
   counts eight unowned `psutil.virtual_memory()` sites; this module is meant
   to become their owner, not the ninth.

**Still open after cut 1:** M1–M8 unchanged; the E1/E2/E3 reconciliation
(deferred to cut 4 as designed); `expert_stats.py` still wired and empty.
