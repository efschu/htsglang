# DESIGN #631 Route A: PP=3 prefill -> TP=3 decode, one world, layout flip

Status: weights decision FIXED (section 1); transition design in progress.
Scope per operator correction 2026-08-07T17:35Z: ONE server instance, ONE
NCCL world, THREE ranks on three cards. Phase A runs PP=3 for prefill,
phase B reshards the SAME ranks to TP=3 for decode. Flip granularity is
regime-wide, never per request. NOT PD-disaggregation, NOT two coexisting
groups, NOT owned_ordinals.

## 0. Measured inputs (all numbers from real boots, #625 battery 2026-08-07)

Sources: /spinning/gpu-battery-results/2026-08-07_625/{RESULTS.txt,
pp3_facts.txt}, /spinning/w625_pp3.boot.log, /spinning/w625_tp3.boot.log.
Model Qwen3.6-27B-INT8-W8A8 (29.10 GiB safetensors), ctx 65536, kv fp8_e4m3,
hybrid: 64 layers = 16 full-attention + 48 linear-attention (GDN).

Cards and host links (measured/runbook):

| GPU | card | NVML total | PCIe | H2D |
|-----|------|-----------|------|-----|
| 0 | RTX 5090 | 32607 MiB | x4 | ~6.4 GB/s |
| 1 | RTX 3080 | 20480 MiB | x8 | ~13 GB/s |
| 2 | RTX 3080 | 20480 MiB | x8 | ~13 GB/s |

GPU<->GPU (through host, no P2P on this rig): 5.1-9.1 GB/s.

Per-rank resident weight bytes, from `Load weight end` lines:

| rank | PP=3 (layers 32/16/16) | TP=3 uneven (auto [28107,17780,17780]) |
|------|------------------------|----------------------------------------|
| 0 (5090) | 14.71 GB | 12.37 GB |
| 1 (3080) | 6.65 GB | 8.58 GB |
| 2 (3080) | 9.02 GB (last stage: +lm_head) | 8.27 GB |

Motivation (same battery): PP=3 prefill beats TP=3 by 2.38x/5.00x/5.22x at
2048/8192/32768 tokens (saves 0.63/4.38/19.48 s per prefill); PP decode is
~3.6x WORSE than the TP+NEXTN production arm (31 vs 112 tok/s). Prefill
wants PP, decode wants TP; hence the flip.

## 1. The weights decision: MOVE-AT-FLIP through a boot-allocated shared arena

The briefing's "both 1/3 of the bytes" is an approximation; the real split
(above) is ~48/22/30% under PP and ~43/29/28% under uneven TP.

### Option "resident both" -- REJECTED by arithmetic, not judgment

Per-card sum of both layouts: 27.08 / 15.23 / 17.29 GB against 31.84 /
20.0 / 20.0 GiB card totals. What must still fit on top: TP decode pools as
measured (KV 4.68/2.66/2.66 GB + mamba 0.51/0.36/0.29 GB), a PP prefill KV
pool, GDN prefill scratch (measured 63 MiB/layer -> ~2 GB on the 32-layer
rank at chunk 2048), CUDA context ~1 GB, and the hard 1024 MiB/card user
reserve. Rank 0: 27.08 + 5.2 + ~2 + ~1 + 1 = ~36 GB > 31.84. Rank 1:
15.23 + 3.0 + ~1 + ~1 + 1 = ~21 GB > 20. Dead on every card.

### Option "hybrid" (3080s resident-both, 5090 reloads) -- REJECTED

Flip latency is bounded by the slowest reload, and that is the 5090 on x4
in every hybrid. The hybrid saves zero flip time and costs the 3080s
6.6-9.0 GB of pool capacity. Strictly dominated.

### Option "re-slice from peers" (TP shard of layer L p2p'd from its PP
### owner) -- REJECTED

No P2P on this rig; cross-GPU runs 5.1-9.1 GB/s through host, no faster
than a host reload, and adds quantized-weight repack coupling (INT8
compressed-tensors shard layout differs from the whole-layer image).
Host reload is simpler and the same speed or better.

### Option "move-at-flip via shared arena" -- CHOSEN

* One VRAM weights ARENA per rank, allocated once at boot, sized
  max(PP bytes, TP bytes) = 14.71 / 8.58 / 9.02 GB. Extra VRAM vs
  TP-only serving: +2.34 / +0.00 / +0.75 GB. That is the whole VRAM price
  of dual-layout capability.
* Both layouts are loaded once at boot (sequentially, through the normal
  loader), snapshotted to HOST RAM, and the device tensors of BOTH
  view-sets are rebound to fixed offsets inside the arena.
* HOST RAM LEDGER (rig: 98 GB total, NO swap; any runtime guard reads
  /sys/fs/cgroup/memory.stat -- /proc/meminfo is lxcfs-falsified in this
  container). Two variants, priced:
  - Variant A, both snapshots pinned: 21.4 / 15.2 / 17.3 GB per rank,
    ~54.4 GB pinned total. Simplest flip (pure H2D refill), but 54 GB
    pinned against 98 GB no-swap alongside the ~28 GB model page cache,
    OS, and any HiCache host tier is aggressive.
  - Variant B, ONE host buffer per rank sized max(PP,TP) = 14.7 / 8.6 /
    9.0 GB, ~32.3 GB pinned total: the buffer always holds the INACTIVE
    layout; the flip streams the dying layout D2H into the buffer while
    the new layout streams H2D out of it, chunk-ordered (D2H chunk i
    before H2D chunk i into the same host offsets), on two streams. PCIe
    is full duplex, so total ~ max(D2H, H2D) + one chunk = 2.0-2.4 s on
    the binding 5090 -- near Variant A latency at 22 GB less pinned RAM.
  Order of work: implement Variant A first (simple, correct, and the
  hermetic layer is identical for both); Variant B is the named follow-up
  behind the same interface, selected at boot by a cgroup-ledger check
  (refuse-loudly if neither fits, never silently unpinned).
* A flip is ONE contiguous H2D refill of the arena from the other
  layout's host snapshot, then switching the active view-set:
  - PP -> TP: 12.37 GB / 6.4 GB/s = **~1.9 s** (5090 binds; 3080s finish
    in 0.6-0.7 s in parallel).
  - TP -> PP: 14.71 GB / 6.4 GB/s = **~2.3 s**.
* CUDA-graph constraint holds BY CONSTRUCTION: TP weight tensors live at
  fixed arena offsets that never change after boot, so decode graphs
  capture once and are never recaptured; the flip only rewrites bytes
  under the same addresses. The PP phase runs eager prefill (prefill
  graphs disabled in the measured recipe), so the PP view-set aliasing
  the same arena bytes is safe -- only one view-set is ever live, and
  the flip protocol guarantees a full refill before the other side runs.

### Break-even (why regime-wide granularity is also the economic answer)

Flip round trip ~4.2 s (1.9 + 2.3) + KV/GDN redistribution (sub-second,
section 2 pricing). One 8k-token prefill saves 4.38 s under PP: break-even
at a single 8k prefill per regime cycle; a 32k prefill saves 19.5 s; queued
prefill batches amortize further. Below ~8k pending prefill tokens the
server simply stays in TP -- the regime gate prices this with exactly the
numbers above.

## 2. KV and GDN state at the flip (design, in progress)

Two state families change ownership axis at PP -> TP:

1. Full-attention paged KV (16 layers). PP owns by LAYER (8/4/4 layers,
   all tokens, all heads); TP decode owns by DCP token vector [30,17,17]
   (all 16 layers per rank, token-sharded). Measured cell size: K = V =
   exactly 1 KiB/token/layer fp8 (PP0 log: 3.46 GiB / (8 layers x 453782
   tokens)), so 32 KiB/token group-wide. ROW FORMAT IS BYTE-COMPATIBLE
   across the flip: the fork's weighted DCP replicates KV heads, so both
   the PP row (full heads) and the TP row (full heads) are the same bytes
   -- the move is a pure row redistribution keyed by (stage layer map,
   token owner rule), no head slicing. Crossing fraction under stage
   split 2,1,1 vs vector [30,17,17]: 1 - (0.50*0.469 + 0.25*0.266 +
   0.25*0.266) = ~63%. A 32k prefix: 1.0 GiB total, ~650 MB crossing,
   ~0.15-0.3 s at 5-9 GB/s host-routed NCCL p2p.
   Payload structure fits the #297 envelope: token ownership is
   layer-independent, so a (sender stage, receiver rank) payload is the
   same slot set repeated over the stage's layers -- the existing
   pack-per-layer/concat convention, generalized by a per-rank layer map
   that both ends derive from the replicated PP partition.
2. GDN conv/ssm state, per live request per linear layer. PP owns whole
   layers (24/12/12, all heads); TP shards HEADS per layer across all 48.
   Measured ~72-76 MB per request summed over all layers (both boots
   agree: PP (5.98+2.99+2.99) GiB/169 slots, TP (0.49+0.35+0.28) GiB/15).
   This move slices WITHIN rows (layer-axis -> head-axis) and needs its
   own packing, but at 4 live requests it is ~300 MB, ~0.1 s.

Draft-model (NEXTN/MTP) weights for the decode arm are small, TP-shaped
only, and stay permanently resident outside the arena -- zero flip cost;
the PP phase simply does not run speculation.

NVML DRIFT WARNING: between 2026-08-07 and 2026-08-08 the rig's NVML
enumeration moved the 5090 from index 0 to index 1. Every boot recipe and
test in this design resolves cards by UUID/name at runtime; a numeric
index in any log above is only valid for the boot that logged it.

The actuator is the #297 kv_reshard envelope (consensus-first bounded MIN
reduction with (armed, ready, epoch, vector); reads-before-writes;
pools pre-sized at boot; NCCL p2p batch exchange). PP->TP adds a new
TRANSITION AXIS (layer-partition -> token-partition) on the same envelope:
build_transition gains a sibling that derives per-rank send/recv row lists
from (live slots, PP layer map, TP token vector) as a pure replicated
function, exactly like the DCP-vector case. Both phases' pools are
allocated at boot (the PP pool and the TP pool coexist; the flip moves
bytes between them), so no growth, no address change, no recapture.

### Envelope invariants carried over literally (#297)

The transition reuses the KvReshardRuntime consensus discipline unchanged:
consensus-first bounded MIN-reduction with (armed, ready, epoch, vector),
equality-checked epoch/target with loud KvReshardError on mismatch;
reads-before-writes with the aliasing falsifier; both layouts' pools and
the weights arena pre-sized at boot -- no growth, no address change, no
CUDA-graph recapture. The PP->TP remap is a NEW owner mapping (layer-axis
-> token-axis) fed to the SAME envelope. Anything that seems to require a
second envelope or a second world is out of scope by construction.

### In-flight prefixes: correctness obligation

Requests whose prefill completed (or is mid-chunk) under PP and whose
decode continues under TP keep their ENTIRE prefix KV across the flip:
every full-attn KV row written by PP-prefill is redistributed to its TP
owner (the <=1 GB / 32k number above IS this move), and every GDN state is
explicitly moved (never assumed to ride along -- the #212 Store-Route
lesson: Mamba state truncates silently if treated as freight). The
acceptance test is token-exact equivalence: decode after
PP-prefill+flip must produce the same tokens as the no-flip reference
decode of the same prompt at temperature 0 within the fork's determinism
envelope. Chunk-boundary rule: the flip only commits at a group-idle
boundary, so "mid-chunk" states do not exist at commit time -- the
ready_fn refuses until scheduled work drains.

### Regime gate

Flips are regime-wide, driven by the regime controller with HYSTERESIS:
the gate prices pending prefill volume against the measured flip cost
(~4.2 s round trip + state move) and must not thrash on mixed workloads;
enter-PP and enter-TP thresholds are separated (enter-PP only above
~8k-16k pending prefill tokens, return-to-TP only when the prefill queue
has been below the floor for a hold-down window).

### Falsifier-first test obligations (before any GPU claim)

1. Aliasing falsifier for the new mapping: a deliberately overlapping
   old/new row layout must fail red without the reads-before-writes
   ordering, green with it.
2. Epoch/vector mismatch: two mock ranks armed with different targets
   must raise the loud desync KvReshardError on both, never proceed.
3. Owner-mapping can-fail: a deliberately broken layer map (wrong stage
   boundary) must fail the transition-builder equivalence check, proving
   the checker can fail.
4. Byte-equivalence harness: simulated PP-layout pools on N mock ranks,
   flip, reassemble -> byte-identical to a directly-built TP layout.

## 3. Architecture (fixed after code survey, file:line refs are tree facts)

### 3.1 One world, two group sets, built eagerly at init

World stays 3 processes. Beside the phase-primary groups, a SECOND group
set over the identical rank list [0,1,2] is built AT INIT on all ranks
(precedent: `_DCP_SPILL` and `_PDMUX_PREFILL_TP_GROUP`,
`parallel_state.py:3025-3089`, including the stated rule that a group
CREATE is itself a collective -- never lazy). `initialize_model_parallel`
refuses two topologies (`world==tp*pp` check at `:2980`, `assert _X is
None` guards), so a new entry point builds: PP group (tp=1, pp=3) and TP
group + DCP group (tp=3, dcp [30,17,17]). Phase routing follows the
`_DCP_SPILL_ACTIVE` flag precedent (`parallel_state.py:2640-2648`) plus
`get_parallel().override(...)` contextvar scoping (`runtime_context.py:
174-211`) -- per-phase code runs under its own geometry scope; no
parallel_state global is mutated at flip time.

### 3.2 Two runner stacks, one scheduler

Per rank: a PP-shaped ModelRunner (built under pp geometry scope, real
layers only for its stage window via `make_layers`/`get_pp_indices`) and
a TP-shaped ModelRunner (+ NEXTN draft runner, TP-only). Precedent for
N runners in one process: `tp_worker.py:385/:440` (target+draft) and the
dual-group lane build under `lane_geometry_override`
(`dual_group_lane.py:5874`). Graph memory pools, flashinfer workspaces,
forward context and the TP-ratio plan are already lane-/context-keyed
(#274); the flip reuses those keys. The attn_dcp_size silent-1 hazard is
closed by construction order: groups exist before any backend constructor
caches `dcp_enabled` (briefing section 2), asserted at build.

The scheduler is ONE object; batches, queues, radix tree and req_to_token
hold GLOBAL slot ids (layout-independent, the #297 no-metadata-rewrite
property). Flip-affected snapshot state is rebuilt at cutover: frozen
`ParallelState` dataclass (`scheduler.py:499`), cached `tp_group`/
`pp_group` handles (`scheduler.py:1269-1276`), `pp_max_micro_batch_size`
(`:1261`), PP loop arrays (`init_pp_loop_state`,
`scheduler_pp_mixin.py:640`). `adjust_hybrid_swa_layers_for_pp`
(`model_runner.py:1493`) mutates model_config in place -- each runner
gets its OWN model_config copy instead. Event-loop dispatch
(`scheduler.py:6889`) reads pp_size once; the flip exits the current loop
to a re-dispatching wrapper.

### 3.3 Weights arena mechanics

Load each layout normally at boot, then PACK: copy every checkpoint
parameter to its assigned fixed offset in the arena (preserving STRIDES
-- compressed-tensors INT8 finalizes `weight` as a transposed view,
`compressed_tensors_w8a8_int8.py:151/:159`), rebind `param.data` to the
arena view (`.data =` rebind precedent: `dual_group_lane.py:769`;
in-place-refill precedent: `hibernate.py:687`), free the originals, and
snapshot the packed arena D2H as ONE contiguous host image per layout.
A flip is then a single contiguous H2D of the other layout's image.
Non-checkpoint device params (marlin workspace,
`marlin_utils.py:861-879`) are excluded from the arena and persist.
GDN modules capture Parameter OBJECTS at construction
(`dual_group_lane.py:739-741`) -- rebinding `.data` is safe, replacing
Parameter objects is not; the pack walk must rebind, never replace.
Decode CUDA graphs bake weight ADDRESSES at capture
(`decode_cuda_graph_runner.py:1682`); arena offsets are fixed for
process life, so graphs stay valid; contents at replay time are correct
because the flip protocol completes the refill before the TP loop runs.

### 3.4 KV and GDN transition on the #297 envelope

Old ownership is per (layer, token): stage_of(layer) owns ALL tokens of
its layers, pool row = global slot id. New ownership: dcp_owner(token)
owns ALL 16 full-attn layers of its tokens, pool row = weighted compact
row. The new plan builder derives, per rank, per-peer payloads = same
token set repeated over the sender's stage layers (token ownership is
layer-independent) -- pack/exchange/checksum/write phases and both
channels (`default_collective_min` over the CPU group,
`_dist_exchange` NCCL p2p batch) are reused as-is; the executor
generalizes only the per-layer row enumeration via a replicated layer
map. Mamba: export/import per-slot blobs (`MambaPool.export_state_blob`
/`import_state_blob`, complete and bit-identical by contract,
`memory_pool.py:956-1009`) inside the graph-safe
`GdnSlotExecutor.between_ticks()` window; head-shard slicing reuses
`get_conv_subblock_spec`/`get_conv_transfer_segments`
(`memory_pool.py:1213+`) -- the q|k|v sub-blocks shard independently,
never as a flat slice.

### 3.4a KV pool residency: TWO RESIDENT POOLS (option priced 2026-08-08)

Operator challenge resolved with arithmetic. Per-rank KV cost at 2
KiB/token/full-attn-layer (fp8, measured): PP pool 16/8/8 KiB/token
(8/4/4 layers), TP pool 15/8.5/8.5 KiB/token (16 layers x [30,17,17]/64)
-- near-equal per rank by construction (stage ratio ~ vector ~ VRAM), so
both-resident ~ 2x: 31/16.5/16.5 KiB/token.

Ledger per rank (MiB): arena 15063/8786/9236; GDN slots both layouts
(16 each) ~1146/690/690; activation+runtime+graphs 4016 (derived demand,
known-conservative -- the falsified-high heuristic); NCCL for TWO group
sets ~500 (UNMEASURED, must be measured at first dual-group-set boot);
CUDA context 900; user reserve 1024. Fixed totals 22649/15916/16366
against 32607/20480/20480. KV budgets 9958/4564/4114 -> global token
ceilings ~329k/283k/255k. BINDING: 3080 rank 2 at ~255k global tokens =
~3.9 concurrent 65536-ctx requests, ~22% below the single-layout
production cap (327700, itself the mamba cap of 4 x 65536 + headroom).
Not halved -- the #625 arms sized pools into luxury free VRAM.

DECISION: two resident pools (option a). The ~22% ceiling price is the
KV analog of the arena's +2.34 GB and is recoverable: the 4016 activation
term is known-conservative and the 500 NCCL term is a guess -- both are
measure-then-reclaim items at first boot. The aliasing-falsifier
exemption in phase_flip_runtime.py is TRUE under this architecture
(source and destination pools are disjoint buffers). Option (b) -- one
shared byte region viewed as either layout, in-place exchange -- would
recover most of the 2x KV factor but re-imports the #297
reads-before-writes hazard across DIFFERENT per-layer structures; it is
the named capacity follow-up behind the same KvPoolView interface, and
the #297 aliasing falsifier TRANSFERS to it if it is ever built. Option
(c) (rebuild a pool at flip) violates no-growth/no-address-change and
the decode graphs' baked addresses; rejected outright.

### 3.4b Step-6 blocking obligations (operator, 2026-08-08)

1. The two UNMEASURED ledger terms -- NCCL-two-group-sets (~500 MiB
   guess) and activation/runtime (4016 MiB, known-falsified-high) -- are
   MEASURED on the first dual-group-set boot and the 3.4a ledger updated.
   A first boot that does not produce these two numbers is incomplete.
2. The 1024 MiB/card user reserve is the hard corridor rule: at first
   boot, NVML-free >= 1024 MiB is verified CONTINUOUSLY under load
   (100-ms sampling, time-series minimum), never as a boot snapshot.
3. The GDN-both-layouts ledger term (~1146/690/690 MiB) is pinned by a
   real-config arithmetic test (Qwen3.6-27B constants: conv_dim 10240 x
   3 cols, temporal 48 heads x 128 x 128, bf16, 48 linear layers split
   24/12/12), not left derived (the M2 laptop lesson: mamba sizing
   killed a spec arm silently).

### 3.5 The flip quiescence predicate is NOT #297 fully-idle

`is_fully_idle` requires an empty waiting queue -- it can never fire
between a request's prefill and its decode, which is exactly when the
flip must run. The flip's ready predicate: no forward in flight, no
partial chunk (`chunked_req is None`), PP micro-batches drained
(`_pp_microbatches_drained`, `scheduler.py:5805`), overlap result queue
empty -- with running/waiting requests PARKED. Consequence: the live
slot set is tree_cache values UNION the req_to_token rows of parked
requests, not the tree alone. Both facts are replicated, so the
transition stays a pure function of replicated inputs.

### 3.6 Flip commit sequence (every rank, same round)

1. consensus MIN-reduction (armed, ready, epoch, target-layout id) --
   loud desync on mismatch, uniform hold on skew (envelope unchanged);
2. full-attn KV move PP-pool -> TP-pool (pack/exchange/checksum/write);
3. GDN blob export -> exchange -> head-sharded import;
4. weights arena refill H2D (overlappable with 2-3: different engines);
5. cutover: set dcp vector + refresh owner bounds (#297 cutover fn),
   swap active runner stack, rebuild ParallelState + cached handles,
   clear/init PP loop state, epoch++, exit to loop re-dispatch.
TP -> PP is the mirror (decode-written rows redistributed to stage
owners). Only the write phases are no-return regions, as in #297.

### 3.6a Operator pins (2026-08-08, hard)

1. GROUP CREATION ORDER: both group sets are created in one fixed,
   documented sequence, identical on every rank (group creation is a
   collective; rank-order divergence is the #431/#616B/#645 crash
   family). The sequence is pinned by a test, not by convention.
2. GRAPH ASYMMETRY: ONLY the TP stack captures decode CUDA graphs. The
   PP stack captures NO graphs at all -- it runs eager prefill exactly
   like the measured #625 recipe (prefill graphs disabled), and PP
   decode never runs in steady state (3.6x worse, the whole point of
   the flip). Price correction: the quoted +2.34/+0.00/+0.75 GB is the
   WEIGHTS ARENA delta only. Second-stack additions: zero graph pools
   (no PP graphs), plus attention workspaces. Preferred: ONE flashinfer
   workspace set shared by both stacks, sized to max(both) -- phases are
   mutually exclusive under the flip protocol, and scratch content never
   survives a forward; the lane-keying hazard (#274) was about
   CONCURRENT forwards, which the flip forbids. If audit of
   capture-time workspace-address binding falsifies sharing, fall back
   to a second workspace set at ~+0.3-0.5 GB/rank -- still within the
   3080 ledger of section 1. PP-only activation scratch (GDN 63
   MiB/layer at chunk 2048) exists only while the PP phase runs and is
   part of the PP-side activation budget, not a new term.
3. ROW BYTE-COMPATIBILITY IS TESTED, NOT ASSERTED: a hermetic test
   builds both layouts' full-attn KV row schemas from the real model
   config (heads, head_dim, dtype, replication rule) and asserts
   byte-layout equality; it fails red if the weighted-DCP head
   replication rule ever changes.
4. PARKED REQUESTS vs CLIENT DISCONNECT: a client vanishing while its
   request is parked mid-flip must not fire any abort path during the
   no-return write phases, must not leak the parked slot, and must not
   desync the quiescence consensus. Abort delivery is deferred until
   after cutover (aborts drain in the first post-flip round). Can-fail
   test: disconnect a parked request during a simulated flip, assert
   clean unpark/cleanup on all mock ranks.

### 3.7 Guards

Refuse to arm the flip with: disk/hierarchical cache ON in the PP phase
(#630 wedge), PD disaggregation, kv-session-offload, weightless-KV,
dual-group lane, spec active in PP phase. #633's PP weight-update-group
deadlock fix is an ancestry fact of this branch -- assert its presence
in the boot check rather than assuming.

### Test inventory (one glob does NOT cover the family)

- test/registered/scheduler/: test_phase_flip_plan.py,
  test_phase_flip_runtime.py, test_weights_arena.py,
  test_gdn_flip_plan.py (+ the #297 test_kv_reshard.py regression).
- test/registered/unit/distributed/: test_phase_flip_groups.py (real
  3-process gloo world -- lives with the other multi-process
  distributed tests on purpose; note the split when running the family).

## 4. Implementation order (each step lands with its falsifiers first)

1. `flip_plan.py`: pure PP<->TP transition arithmetic (layer map x token
   vector), byte-identity harness on mock per-layer pools + falsifiers
   (broken layer map must fail red; aliasing; coverage-exactly-once).
2. Envelope generalization: PhaseFlipRuntime sharing the #297 consensus/
   checksum/exchange helpers; desync + abort-with-pool-untouched tests.
3. Weights arena: pack/rebind/image/refill on CPU tensors hermetically
   (stride preservation, alias preservation, non-checkpoint exclusion).
4. GDN blob flip packing (sub-block segments) + bit-identity tests.
5. Dual group-set init entry + dual-runner boot + scheduler flip
   protocol + regime-gate wiring (integration; first GPU boots).
6. Rig validation: PP=3 prefill -> flip -> TP=3 decode, token-exact vs
   no-flip reference, measured flip cost vs the section 1 budget.

   MEASUREMENT DOCTRINE (standing user order, 2026-08-08, applies to
   every decision-feeding number in this step): all timings are
   MS PER ROUND PER RANK, split into COMPUTE vs WAIT, taken with
   CollectiveClock (#252 -- 0.13% overhead, graph-replay-honest).
   Never tok/s as a decision basis. Specifically:
   - prefill before/after: ms/Prefill per rank compute+wait (the #625
     baseline is already in this form -- comparisons stay in it);
   - decode before/after: ms/Verify per rank compute+wait, with accept
     length and verify_ct recorded alongside EVERY decode round;
   - flip cost: per-phase ms on every rank (KV pack/exchange/write from
     the runtime stats, arena refill, GDN move, cutover), reported per
     rank so the binding rank is identified, not averaged away;
   - the two unmeasured ledger terms (NCCL two-group-sets, activation)
     measured on the same boots. The NCCL term has an EXISTING
     instrument: mem_ledger/nccl_probe.py's measure_communicator_init
     already brackets every GroupCoordinator pynccl constructor,
     including the secondary flip set's -- arm it on the first
     dual-group-set boot and read both sets' real buffer cost from it
     (the secondary set's communicators live for the whole boot, through
     both phases);
   - every run >= 10 s, and an A-vs-A same-boot noise floor is
     established BEFORE any delta is read (the 0.10% #625 floor is the
     reference shape);
   - the 8k-prefill break-even line of section 1 is verified in these
     units (flip round-trip ms vs PP-vs-TP ms/Prefill delta at 2k/8k/32k).

## 5. Integration wiring plan (remaining step-5 slices, file:line from
## the 2026-08-08 survey; each slice lands with tests before the next)

STATE 2026-08-08 (session 2): 5.2 DONE -- TP-stack boot builder landed
(managers/phase_flip_boot.py: server-args derivation, flip geometry
scope, snapshot/free/pack/image weights choreography, TP pools + decode
graphs under the scope, PP rebind + refill; scheduler hook at the end of
init_model_worker, before the post-capture resize). 108 green tests
(88 prior + 20 new in test_phase_flip_boot.py). Desk-written GPU
integration validated at step 6. NEXT: 5.3 scheduler flip protocol.
Pin status: 1 discharged (manifest), 2 discharged STRUCTURALLY
(is_phase_flip_pp_stack carve before the harmonization collective +
poison gates on both capture entry points, hermetically pinned red and
green), 3 DISCHARGED (boot assert assert_row_schema_compatible on the
real pools + hermetic real-config test with the head-sharded red arm),
4 discharged at unit level (scheduler-level replay in 5.3).

SURVEY CORRECTION to 3.1 (load-bearing, found in session 2): forward-time
collectives reach groups through the parallel_state MODULE GETTERS
(tensor_model_parallel_all_reduce -> get_tp_group()), NOT the
runtime_context contextvar; the lane precedent never exposed this because
tp_size=1 short-circuits the collective (linear.py:2338). The contextvar
override alone would therefore shard weights correctly and then
all-reduce over the primary tp=1 group -- a silent no-op. Resolution:
_PHASE_FLIP_TP_ACTIVE module routing in get_tp_group /
get_attn_tp_group / get_dcp_group / get_pp_group (the
_ENABLE_PDMUX_P_TP/_DCP_SPILL_ACTIVE precedent), toggled rank-uniformly
at the boot build scope and at cutover; activation without built flip
groups REFUSES loudly. Both mechanisms together are the geometry scope
(phase_flip_boot.phase_flip_tp_scope).

5.3 DONE (session 2): scheduler flip protocol landed --
build_phase_flip_runtime factory (channels over the flip_tp world-spanning
set, both pool views, landed helpers, pre_cutover = GDN guard + arena
refill), on_round hook beside the #297 block raising PhaseFlipLoopExit
after commit, production cutover with the FULL rebuild list closed by a
verify_flip_cutover completeness self-check (every rebuilt reference
checked against the routed source of truth as the cutover's last step;
red arms prove one stale handle/ps/worker/window is caught loudly),
abort deferral through the real Scheduler.abort_request router
(arm_phase_flip activates the window, refused arm drains immediately),
maybe_sleep_on_idle flip-pending skip, run_phase_flip_event_loops
re-dispatch wrapper. Pin 4 discharged at scheduler level: real router ->
real window -> real runtime threads -> real production cutover, aborts
apply in order with the TP stack already active. GDN mover is a LOUD
REFUSAL placeholder (flip with live requests refuses; empty flip works)
-- slice 5.3b implements the real mover before the one-request rung.
122 green tests; canonical family runner: scripts/run_631_flip_family.sh.

5.3b DONE (session 2): production GDN mover landed
(managers/gdn_flip_mover.py) and wired as the flip's first pre_cutover
leg. DEVIATION from 3.4's blob route, deliberate: the mover works on the
conv/temporal pool tensors directly with the gdn_flip_plan primitives
(the landed, tested route) instead of export/import_state_blob -- the
blob's extra fields are covered by REFUSAL preconditions instead
(ReplaySSM ring buffers, int8 mamba checkpoint pool, multi-conv layouts
all refuse loudly; V1 scope). The reachable-refusal contract holds by
construction: gdn_flip_preconditions re-validates per flip and checks the
plan-derived shard spec against the TP pool's ACTUAL tensor shapes.
POOL-SHAPE FIXES uncovered while wiring (would have been silent #345
corruption, caught at desk): the TP stack rode is_draft_worker into
draft-shaped pools -- [0]-layer full-attn list, head-sharded KV via
draft_pool_is_replicated, and the shared req pool's PP-shaped mamba pool.
Fixed with is_draft_pool_worker (pool-geometry sites only) and by giving
the TP stack its OWN HybridReqToTokenPool (TP-shaped mamba) while
sharing the request-mapping TENSORS by rebind (req_to_token,
req_index_to_mamba_index_mapping; slot spaces asserted equal). 135 green.

5.4 DONE (session 2): the flip is the #363 actuator's THIRD axis
(regime_act.py: phase_flip_arm + current_phase_fn injectables, regime ->
phase mapping prefill_heavy->pp / decode_heavy->tp, direction =
_DIR_OF_PHASE[current]). NO new gate: thresholds, sustain and dwell
hysteresis are the classifier's existing machinery (RegimeSensor /
DwellGate / act interlocks / evidence-gated act mode -- all untouched).
Unwired axis = byte-identical #363 (pinned). The section-1 break-even
number (~8k pending prefill tokens for enter-PP) is a classifier
threshold TUNING item for step 6, not new machinery. STEP 5 IS COMPLETE:
206 green tests via scripts/run_631_flip_family.sh (now incl.
test_regime_act.py). NEXT: step 6 GPU validation per 5.5 and the
measurement doctrine of section 4, after the crashfix defect-1 merge and
a card window (operator sequences both).

Step-6 watch items added in session 2: (a) the uneven plan + cp token
vector are process-globals installed AFTER the primary PP build --
PP-phase code paths must never consult them with a non-rank-local size;
(b) get_server_args() readers at forward time see the PRIMARY args in
both phases (the TP stack's copy is published only during its build);
(c) TP-stack pool sizing uses the copied mem_fraction_static -- the
3.4a ledger is enforced by measurement at first boot, not yet by code.

### 5.1 Server args surface
- New flags: --phase-flip (off default; gates EVERYTHING below),
  --phase-flip-tp-vector (the decode DCP vector, e.g. 30,17,17),
  reuse --pp-stage-ratio/--pp-layer-ratio for the prefill split.
- Validation in a _handle_phase_flip: refuse with PD, hierarchical
  cache, dual-group lane, dp/ep > 1; require pp_size == world == vector
  length. Early and loud (server_args.py handler family).

### 5.2 Dual-runner boot (tp_worker/model_runner)
- Boot as the PP topology (pp_size=3, tp=1) exactly like the measured
  #625 recipe. After the primary stack is up, call
  initialize_phase_flip_secondary_groups(tp_size=3, pp_size=1,
  dcp_size=3) -- eager, before ANY backend constructor caches dcp state
  (the attn_dcp_size silent-1 hazard; assert group formed).
- Build the TP-shaped ModelRunner (+ NEXTN draft) under
  get_parallel().override(tp_group=flip_tp, dcp fields, tp_size=3,
  pp_size=1) -- the lane build precedent (dual_group_lane.py:5874,
  lane_geometry_override); each runner gets its OWN model_config copy
  (adjust_hybrid_swa_layers_for_pp mutates in place,
  model_runner.py:1493).
- Weights: load PP layout, plan+pack arena (weights_arena.py), image;
  load TP layout THROUGH THE SAME arena (pack, image); leave the boot
  phase's image live. Non-checkpoint params excluded (marlin workspace,
  marlin_utils.py:861-879). Draft (NEXTN) weights TP-only, OUTSIDE the
  arena, permanently resident.
- Pools: PP stack allocates its stage pools (existing PP path); TP
  stack allocates full-attn TP pool + TP mamba pool under the 3.4a
  ledger. Decode CUDA graphs captured ONCE on the TP stack after its
  arena image is live (pin 2: PP stack captures NOTHING).
- Real-config row-schema pin (operator pin 3) lands here: both pools
  exist in one process -> assert per-token row byte-layout equality
  from the actual configs, red if head replication changes.

### 5.3 Scheduler flip protocol
- Scheduler holds active_stack ("pp"|"tp"), both worker stacks, and the
  PhaseFlipRuntime built by a build_phase_flip_runtime(scheduler)
  factory (mirror build_kv_reshard_runtime, kv_reshard.py:697):
  channels = default_collective_min(tp_cpu_group of the WORLD-spanning
  set) + _dist_exchange over the flip_tp device group; views = both
  pools; live_slots/ready/guards = the landed helpers; pre_cutover_fns
  = [arena refill H2D, GDN blob move]; cutover_fn = production cutover.
- on_round hook next to the kv_reshard hook (scheduler.py:4186 block);
  AbortDeferralWindow activated at arm, drained after cutover/disarm;
  abort handling routed through window.submit while active.
- Cutover rebuilds: ParallelState frozen dataclass (scheduler.py:499),
  cached tp_group/pp_group handles (scheduler.py:1269-1276),
  pp_max_micro_batch_size (:1261), init/clear PP loop state
  (scheduler_pp_mixin.py:640), set_cp_token_ratios + owner-bounds
  refresh (the #297 cutover, kv_reshard.py:675), then EXIT the current
  event loop to a re-dispatching wrapper around dispatch_event_loop
  (scheduler.py:6889 reads pp_size once -- wrap, do not patch).
- maybe_sleep_on_idle must also skip while phase-flip pending
  (scheduler.py:6820 precedent for kv_reshard.pending).

### 5.4 Regime gate
- regime_act.py gains the flip action beside KvReshardRuntime.arm:
  prefill_heavy + queued_prompt_tokens above threshold -> arm pp
  direction; decode_heavy sustained -> arm tp direction; hysteresis from
  RegimeSensor/DwellGate (regime_classifier.py:273/:478). Evidence-gated
  act mode unchanged (server_args.py:7147).

### 5.4a Cross-strand dependency (#622, recorded 2026-08-08)

The crashfix strand root-caused the rig's async bug to first-token
NO-SYNC in the base sampler (_sync_token_ids_across_tp default-off): on
this mixed-arch rig, ranks can read DIFFERENT first tokens and so
DISAGREE about batch membership (one rank sees EOS, peers do not). The
flip's replicated-live-set assumption is FALSE under the unfixed bug.
Defenses and consequences:
- The consensus envelope + receiver-derived payload sizes make a
  membership disagreement a LOUD refusal, never a silent mixed layout --
  pinned by test_can_fail_batch_membership_disagreement_is_refused_loudly
  (one rank believes a whole request finished -> size-mismatch
  KvReshardError, no hang).
- Step-6 rig validation runs on a tree that INCLUDES the crashfix
  token-sync fix (fix/collective-stream-622 merge, operator coordinates
  the order); soaking on an unfixed tree would make every flip anomaly
  ambiguous.
- Graph asymmetry (pin 2) is STRUCTURAL in 5.2: the PP stack's runner is
  built with decode-graph capture absent (no decode graph runner
  constructed), not merely configured off.

### 5.5 GPU validation order (step 6)
- Boot flip-enabled, NO flip armed: PP phase must reproduce #625 PP
  prefill numbers (A-vs-A floor first).
- Manual /phase_flip RPC (mirror /kv_reshard, http_server.py:1200):
  flip empty -> flip back, measure per-phase ms per rank.
- One request: PP prefill -> flip -> TP decode, token-exact vs no-flip
  reference at temperature 0.
- Then the regime-gated automatic flip under mixed load, ms/round
  compute-vs-wait doctrine throughout.
