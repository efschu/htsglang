# Route A (#631) in production on 30030: bring-up, KV sizing, benchmarks

2026-08-08. Qwen3.6-27B-INT8-W8A8 on the main rig (1x RTX 5090 32 GB,
2x RTX 3080 20 GB, no P2P/NVLink). ONE instance, three ranks: PP=3
prefill (stage ratio 2,1,1) -> live flip -> TP=3 decode (weight shard
30,17,17) with NEXTN speculation. Transport barlink BAR1. Tree
`feat/route-a-631`.

The production stop guard `/spinning/PRODUCTION_STOPPED` was lifted by
operator order for this bring-up; the crash watchdog stays
DECOMMISSIONED (inactive + disabled) and was not re-armed.
`/spinning/COUNTERTEST_NCCL` was cleared deliberately — the barlink-vs-NCCL
counter-test moved into the crashfix strand's own vehicle, so leaving the
toggle armed would have silently booted production on NCCL.

---

## 1. Boot recipe

`scripts/route_a_631_prod_boot.sh`, invoked as:

```
HICACHE=0 RANK_MIB=20800,12500,13600 SGLANG_UNEVEN_TOKEN_VECTOR=13,11,8 \
  bash scripts/route_a_631_prod_boot.sh
```

which launches:

```
--model-path .../Qwen3.6-27B-INT8-W8A8 --trust-remote-code
--served-model-name Qwen3.6-27B
--tp-size 1 --pp-size 3 --pp-stage-ratio 2,1,1
--rank-gpu-id 0,1,2 --rank-gpu-memory-mib 20800,12500,13600
--enable-phase-flip --phase-flip-tp-vector 30,17,17
--disable-overlap-schedule
--kv-cache-dtype fp8_e4m3 --context-length 262144
--max-running-requests 4 --max-mamba-cache-size 20
--speculative-algorithm NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--reasoning-parser qwen3 --tool-call-parser qwen3_coder
--chat-template-default-kwargs '{"preserve_thinking": true}'
--enable-cache-report --enable-metrics --host 127.0.0.1 --port 30030
```

Environment that matters: `SGLANG_BARLINK=1`,
`SGLANG_BARLINK_TRANSPORT=bar1`, `SGLANG_UNEVEN_DCP=1`,
`SGLANG_UNEVEN_DCP_WEIGHTED=1`, `SGLANG_UNEVEN_TOKEN_VECTOR=13,11,8`,
`SGLANG_MAMBA_SSM_DTYPE=bfloat16`,
`SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`,
`SGLANG_COLLECTIVE_CENSUS_INTERVAL=50`, `SGLANG_MAMBA_PIN_TRACE=50`,
`SGLANG_VRAM_FLIGHT_DIR=/spinning/flight_605`.

Cards are resolved by NVML **UUID**, 5090 first, never by index.

### BAR1 aperture budget (new, required)

barlink BAR1 had never been run with the flip: the acceptance boots set no
`SGLANG_BARLINK*` at all, so they ran NCCL. (The `barlink build window`
lines in those logs come from `build_window_enabled()`, which defaults
True independently of the transport — they do not evidence barlink.)

The flip doubles the wire-carrying process groups, and the stock 96 MiB
per-group default does not fit four groups into a 3080's aperture:

```
BAR1 free per NVML 248 MiB - reserve 32 MiB = 216 MiB usable
vs 4 x 96 = 384 MiB requested
-> world:0 96, pp:0 96, flip_tp:0 clipped to 24, flip_dcp:0 clipped to 0
-> Bar1Failed, all three ranks dead at group build (boot 15:26:40Z)
```

Budgeted by the payload each group carries — `pp:0` 96 (chunked prefill
activations, ~20 MiB/hop, the largest payload on the rig), `flip_tp:0` 48
and `flip_dcp:0` 32 (decode-sized), `world:0` 24 — total 200 of 216 MiB.
All four groups then report `ACHIEVED=bar1`.

---

## 2. The KV pool: what was wrong and what it is now

The pool was 15-25x too small. Two defects, both fixed in `512273a38f`.

### 2a. The weights arena was allocated on top of the live TP weights

`phase_flip_boot` step 4 allocated the arena while the freshly loaded TP
weights were still resident, so boot peaked at `originals + arena`
(torch `allocated_peak_bytes`): **29.27 GiB** on the 5090, **16.64** and
**17.81 GiB** on the 3080s. Confirmed against the 100 ms NVML trace: a
single sample at 15:30:59.842 read free `[2304, 3843, 1096]` MiB, matching
the arena size to within 83 MiB on both 3080s.

The peak never recurs at runtime — serving sits ~7 GiB below it — but the
corridor floor is a **continuous** minimum, so that one boot sample forced
`--rank-gpu-memory-mib` down to roughly half of each card, permanently.

Step 2 already did snapshot-then-free for the PP layout; step 4 did not do
it for TP. `image_from_tensors` exists precisely for this and says so.
Since the host image was already built one line later, the fix is a
reordering, not a new allocation: peak becomes `max(originals, arena)`.

### 2b. One vector drove both the weight shard and the KV token split

The flip installed the same vector for `set_tp_partition_ratios` and
`set_cp_token_ratios`. These optimise against different resources — the
weight shard follows COMPUTE, the token split follows each rank's memory
left AFTER its weights land — so the most compute-loaded rank bound the
allocator's min-reduce:

```
rank 0: 12750 tok / ratio 30 = unit  425   <- binds
rank 1: 68646 tok / ratio 17 = unit 4038
rank 2: 30515 tok / ratio 17 = unit 1795
-> global max_total_num_tokens 27200, ranks 1 and 2 left idle
```

The server already computes the token-proportional vector and logs it as a
restart hint, but nothing could act on it because that line overwrote it.
`parse_flip_token_vector` now reads `SGLANG_UNEVEN_TOKEN_VECTOR`; unset, it
returns the flip vector, so the default path is byte-identical.

### 2c. Result

| quantity | before | now |
|---|---|---|
| max_total_num_tokens, PP phase | 46422 | **278104** |
| max_total_num_tokens, TP phase | 27200 | **318176** |
| context length | 65536 | **262144** |
| max_running_requests | 4 requested, **1 effective** | **4** |
| per-rank budget MiB | 16150 / 10550 / 10550 | 20800 / 12500 / 13600 |

`max_running_requests` was being silently reduced 4 -> 1: the auto-sized
GDN/mamba pool gave 5 state slots and the admission ratio (radix cache on)
needs 5 per request. `--max-mamba-cache-size 20` restores a real bs=4.

### 2d. Why it is not bigger yet — the honest ceiling

278104 tokens is ~1.06 sessions of 262144. Plain TP=3 production reached
669440 at the same context. The difference is structural, not tuning:
**both phases' KV pools and both mamba pools are resident simultaneously**
(`phase_flip_boot.py:465-500`: "The pools themselves stay layout-specific;
only the mapping tensors alias"; `kv_reshard.py:21`: "the PP pool and the
TP pool coexist"). Each pool therefore gets roughly half the budget.

Two further ceilings were measured, not guessed:

- **Per-stack budget double-spend.** Each stack calls
  `_profile_available_bytes` with its own `pre_model_load_memory`
  reference, and the TP stack's is taken after the PP pools are already
  resident. Raising rank 0 to 24800 MiB was refused at TP-stack build
  ("5.36 GiB free ... 3.81 GiB short of the budget") even though the 5090
  showed 4 GiB free at idle. The 5090's slack is unreachable until this is
  fixed.
- **Corridor.** At 22300/13500/13500 with vector 13,11,8 the idle minimum
  was `971 / 120 / 851` MiB free — a breach on all three cards. The
  committed budget is the largest that held.

---

## 3. Nenngroessen (read from the server, not assumed)

- **Max context:** 262144 booted; `max_req_input_len` follows the pool.
- **KV pool:** 278104 tokens global (PP phase), 318176 (TP phase).
  Per-rank PP allocation 26362 / 146874 / 67788 rows of full-attention KV
  (K 0.40 / 2.24 / 1.03 GB, V likewise); TP phase 240960-row token split.
- **Hybrid cap:** `max_running_requests x (context_len + 8)` =
  4 x 262152 = **1572912** tokens — no longer the binding term (it was
  262176 at ctx 65536).
- **GDN/mamba slots:** 20 (`--max-mamba-cache-size 20`). Per rank
  conv_state 0.03/0.01 GB, ssm_state 0.74/0.37 GB, intermediate ssm 0.70/0.35 GB.
- **Spill:** **none armed.** Every tier reports absent via the dashboard's
  `/api/live_snapshot`: `kv_session_host_ram` (flag off),
  `hicache_host_ram` and `hicache_file_disk` (not booted with a
  hierarchical cache), `remote_rig_*` (no destination configured).
  `--enable-hierarchical-cache` is **hard-refused** with
  `--enable-phase-flip` (`server_args.py:7230`, citing "#630: PP x disk
  HiCache wedges at warmup"), and `--kv-pressure-ladder auto` refuses on
  this rig because it cannot map ranks to cards on a mixed-model node.
  This is the open item, see section 6.
- **Radix/prefix cache:** on (`disable_radix_cache` False),
  `--enable-cache-report`.
- **CUDA graphs:** decode `full`, max_bs 24, captured bs
  `[1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24]`; prefill graphs disabled.
  `cuda graph: True` observed in every post-flip decode record.
- **Speculation:** NEXTN, 3 steps, eagle-topk 1, 4 draft tokens. Present
  only in the TP phase — the PP scheduler boots `spec_algorithm` NONE with
  no draft worker, which is what makes an accept-length reading a
  structural proof of the phase.
- **Flip policy:** no automatic cadence. Armed by `POST /phase_flip`
  (`{"direction":"pp_to_tp"|"tp_to_pp"}`); commits at the next consensus
  boundary where every rank is quiescent, else abandons after 30 s parked.
  Measured commit: 955 / 1203 / 1678 ms per rank, empty live set.
- **Corridor:** idle `2323 / 2158 / 2075` MiB free; under bench load
  `2255 / 2146 / 2029` MiB. Floor 1024, held continuously.

---

## 4. Benchmarks

Instrument: the original club-3090 `scripts/bench.sh` against
`http://127.0.0.1:30030` (`URL`/`MODEL` overridden; it defaults to 8020 and
a vLLM container name). Quality suites were NOT run. Prefill is measured in
the PP phase and decode in the TP phase — a run reporting both from one
phase has not measured Route A.

### 4a. Prefill, PP=3 phase

`scripts/route_a_631_prefill_ladder.py` — random `input_ids` per draw so
prefix caching cannot contaminate a repeat, `max_new_tokens=1` to isolate
prefill, warm-up discarded, 3 kept draws, median.

| input tokens | median ms | tok/s | draws (ms) |
|---|---|---|---|
| 2048 | 484.4 | **4227.8** | 485.9 / 484.4 / 484.1 |
| 8192 | 1140.3 | **7183.8** | 1140.3 / 1139.7 / 1143.0 |
| 32768 | 4881.4 | **6712.8** | 4795.3 / 4881.4 / 4923.0 |

Within 3-5 % of the acceptance ladder (4335 / 7439 / 7090) despite 4x the
context and ~6x the pool.

### 4b. Decode, TP=3 phase, NEXTN live, bs=1

club-3090 `bench.sh`, 3 warm-ups + 5 measured per prompt class.

| config | class | decode TPS | wall TPS | TTFT | CV |
|---|---|---|---|---|---|
| ctx 65536, small pool | narrative | 79.50 | 78.80 | 112 ms | 2.9 % |
| ctx 65536, small pool | code | 100.18 | 98.17 | 111 ms | 1.3 % |
| ctx 262144, big pool, divergent vectors | narrative | 74.02 | 73.42 | 109 ms | 8.2 % |
| ctx 262144, big pool, divergent vectors | code | 96.44 | 95.14 | 111 ms | 3.5 % |
| **ctx 262144, big pool, vectors fixed** | narrative | **79.02** | — | 112 ms | **0.9 %** |
| **ctx 262144, big pool, vectors fixed** | code | **103.21** | — | 109 ms | 2.7 % |

The last pair is the shipped configuration. Fixing the token-vector
divergence (section 5.5) is worth +6.8 % narrative and +7.0 % code AND
collapses the narrative CV from 8.2 % to 0.9 % -- the wrong owner rule was
also the source of the run-to-run instability. At ctx 262144 with a 6x
pool it now beats the ctx-65536 small-pool config on code (103.21 vs
100.18) and matches it on narrative.

Reference: production's last measured state (2026-08-05, window 10,
barlink **device** sub-transport) was 67.4-68.5 narrative / 88.1-88.6 code.
The big-pool config is **+8 % narrative / +9 % code** against it, and the
small-pool config was +16 / +13 %. The 4-7 % given back for 4x context and
6x pool sits partly inside the narrative CV of 8.2 %.

Content axis matters and is reported per class deliberately: accept length
tracks output content on this rig, so a single decode figure without its
content class is a fiction.

### 4c. Concurrency ladder (client aggregate, TP phase)

`scripts/gpu_battery/s14_decode_punkt.py`, 2048-token context, 6 s ramp,
15 s window, two independent passes = the A-vs-A floor.

| bs | pass A tok/s | pass B tok/s | A-vs-A spread |
|---|---|---|---|
| 1 | 117.33 | 118.27 | 0.8 % |
| 2 | 215.47 | 209.27 | 2.9 % |
| 3 | 279.13 | 263.87 | 5.5 % |
| 4 | 284.33 | 284.80 | 0.2 % |

Scheduler-side records for the same window: `accept len 4.00, accept rate
1.00, cuda graph: True, gen throughput 281-288 tok/s` at
`#running-req: 3` (the s14 probe's repetitive content accepts every draft
position; natural prose in 4b runs 2.85-3.57).

Aggregate throughput scales ~2.4x from bs=1 to bs=4 and flattens between 3
and 4.

### 4d. Phase proof

Required, because an unproven phase makes every decode number meaningless.
`/get_server_info` is NOT a valid witness — it still reports the PP-phase
`tp_size`/pool after a committed flip (named residual: scheduler
token/memory info is not re-read at cutover). The proof used instead is
structural: the PP phase boots with `spec_algorithm` NONE and no draft
worker, so a non-empty `accept len` in a scheduler decode record can only
come from the TP phase. Every decode point above was taken after a logged
`PHASE-FLIP DONE pp_to_tp` with no subsequent `tp_to_pp`, and every decode
record carried an accept length.

---

## 5. Defects found and fixed

1. **barlink BAR1 x phase flip could not boot** — four wire-carrying
   groups x 96 MiB default vs 216 MiB usable aperture. Config fix in the
   boot script, with the arithmetic recorded.
2. **Weights arena allocated on top of live TP weights** (`512273a38f`) —
   halved every rank's VRAM budget for the whole process life.
3. **One vector for weight shard and KV token split** (`512273a38f`) —
   min-reduced the decode pool to 27200 tokens.
4. **`--kv-pressure-ladder auto` refuses on a mixed-model node** — made
   opt-in in the boot script rather than defaulted.
5. **The cutover reinstalled the WEIGHT vector as the owner rule**
   (`f0df756788`). `PhaseFlipStacks` carried one vector; both
   `set_cp_token_ratios(...)` at cutover and the `tp_vector` handed to
   `build_phase_flip_transition` read it — and the latter's own docstring
   calls that argument "the weighted DCP token vector of the TP layout".
   So after every cutover the owner rule and the row-routing plan split
   rows under a vector the pools were not sized under. Latent while the
   two vectors were equal by construction; making the token side
   overridable is what made it reachable, and it is almost certainly the
   mechanism behind defect 6 below. Fixed by carrying both vectors,
   named for the question each answers.

Open, not fixed here:

6. **KV slot OOB under pool exhaustion.** At the 27200-token pool, a
   benchmark run drove `store_kvcache` into its own guard —
   `SGL_DEVICE_ASSERT(index >= 0 && index < size_limit)`,
   `jit_kernel/csrc/elementwise/kvcache.cuh:112` — killing all three ranks
   with a device-side assert. The pool handed back an invalid slot instead
   of retracting. The bigger pool makes it far less reachable but does not
   fix the allocator's behaviour at exhaustion.
7. **Per-stack budget double-spend** (section 2d). Each `ModelRunner`
   reads its own `pre_model_load_memory`, and the TP stack's reading is
   taken after the PP pools are resident, so those bytes are never
   charged: device use is roughly `2*B - W_pp` against a flag value `B`.
   Under a shared arena this becomes mandatory to fix — one allocation
   must mean one budget.
8. **Both phases' pools resident** (section 2d) — the main remaining lever.

---

## 6. Open: reuse and spill across the flip

The requirement: on the PP -> TP switch, reuse what both layouts share and
spill what the inactive layout does not need, so the KV pool becomes big
again.

The weights already work this way — one arena sized `max(both layouts)`
plus pinned host images, refilled at the flip. The KV and mamba pools never
received the same treatment, which is exactly the duplication in 2d.

### 6a. HiCache as the cross-phase carrier: evaluated, rejected

Attractive because `write_through` already pushes every completed node's
KV to the host tier as prefill produces it. It does not work, for two
independent reasons, either alone fatal:

- **It never removes a device pool.** HiCache's model is *device = L1,
  host = L2*. The duplication exists because `phase_flip_boot` calls
  `tp_worker.alloc_memory_pool()` while the PP stack's pools are live;
  no host tier changes that. `max_total_num_tokens` is a DEVICE quantity,
  so the carrier idea moves it by exactly zero.
- **Host tiers are layout-specific and per-process.** `pool_host/mha.py`
  derives `size_per_token` from `device_pool.layer_num` and takes
  `start_layer`/`end_layer` from the writing pool. Under PP those are the
  stage's 8/4/4 layers; the TP pool has 16. Different bytes per token,
  different layer identity. And the three ranks are separate processes:
  PP rank 0's host tier holds layers 0-7 for ALL tokens, while TP rank 0
  needs all 16 layers for ITS ~40 % of tokens, 8 of which physically live
  in another process. Reconstructing that IS the flip, plus a D2H and an
  H2D per byte — against the ~5 ms the device-side move measures today.

The disk tier cannot serve as the carrier either: `hicache_storage.py`
appends `_{pp_size}_{pp_rank}` to the KV key when PP is enabled, so
PP-phase pages are written under a namespace the TP phase (pp_size 1)
never looks up. A clean miss — safe, and useless as a carrier.

Mamba is not the exception one might hope for. HiCache does know about
mamba state (`hybrid_pool_assembler.py` wires a `MambaPoolHost` as a
`PoolName.MAMBA` entry), but `MambaPoolHost.__init__` derives
`num_mamba_layers` and both state shapes from the writing device pool.
The PP mamba pool is *stage layers x full width*; the TP pool is *all
layers x this rank's head shard*. Both axes differ, so it is even more
layout-locked than KV.

What HiCache IS worth: the plain host-RAM L2 that production always ran,
for prefix-hit rate. Its blanket refusal cites #630, whose actual root
cause (`9da9dfd025`, already an ancestor of HEAD) was an unbounded
`work.wait()` in `_drain_async_work` against its deadline-polling sibling,
plus a `torch.distributed.recv` with no timeout — on the write-through
ACK path, which runs whenever hierarchical cache is on, RAM tier or not.
Disk was the configuration it was observed in, not the mechanism. The
gate can be narrowed to `hicache_storage_backend is not None`, but only
after three other sites are closed: the cutover does not invalidate
`cache_controller._dcp_owner_ctx_cache` (the #297 cutover does, and
`dcp_size` changes 1->3 across the flip), the flip's quiescence predicate
carries no HiCache term while `is_fully_idle` does, and the tree cache and
controller are bound to the PP stack's pools for process life.

### 6b. Shared KV/mamba arena: the answer

Sized `max(PP, TP)` rather than their sum, with the inactive layout's
backing released — the same shape as the weights arena.
`DESIGN_631:388-393` already prices this as the named capacity follow-up.

Do NOT overlay two view sets on one flat region: that works for weights
because a flip replaces all bytes from an image, whereas here the flip
permutes live rows in place. Use instead two VA reservations with mutually
exclusive physical backing — `KvVmmBufferOwner` already provides it, and
`runtime_set_backing_tokens` states the needed invariant verbatim:
"Addresses never move, so captured CUDA graphs keep replaying."

One real ordering bug must be fixed first: in `PhaseFlipRuntime._execute`
the local leg reads and writes per layer in one loop, the only place where
a destination write precedes a source read. Harmless with disjoint pools,
fatal under sharing. Hoisting the reads is a two-line change.

Spill, decisively: correctness does not need it. The live set (radix
values union parked requests' rows) is everything the allocator considers
referenced, and the flip already moves it. What remains in the source pool
is by definition unreferenced, so "spill the inactive layout" means
optionally preserving the PREFIX CACHE across a flip, never correctness.
The design is therefore: evict to the protected set before deriving the
transition (which also bounds the transit buffers — `_dist_exchange`
allocates whole payloads on device, fine at today's 7 MiB, GBs at
C~520k), and treat host-side prefix carry as a skippable performance
add-on.

Projected, per the byte model (cost per global token is the SUM of both
layouts today, the MAX when shared):

| step | expected max_total_num_tokens |
|---|---|
| today | 278104 |
| shared KV arena | ~520k (1.9x) |
| + token vector 16,8,8 (makes cTP == cPP per rank) | ~537k |
| + shared mamba arena | ~626k (2.25x) |
| + budget double-spend fix | bounded by 669440, the TP-only reference |

Treat 669440 as the ceiling, not more: the model omits activation reserve,
graphs, NCCL buffers and the continuous corridor minimum at boot peak.

---

## 6c. Shipped after the capacity pass (2026-08-08, commit fa6ceab7b3)

Re-budgeted to take up the corridor slack and, more importantly, to match
the KV TOKEN vector to each rank's MEASURED capacity rather than to the
weight shard. The units the allocator min-reduces over are what matter:

| config | per-rank units | TP pool |
|---|---|---|
| vector 13,11,8 | 9696 / 13058 / 16386 | 310272 |
| vector 25,25,20 | 5426 / 5463 / 4666 | 326620 |
| **vector 28,26,20 (shipped)** | 4601 (min) | **340474** |
| vector 21,24,22, budget +1.1 GiB/card | 7679 / 7451 / 7413 | **496671** -- OOM, corridor breach |

The 496671 boot is the load-bearing measurement: with the vector matched
to capacity the units equalise (7679 / 7451 / 7413) and the pool reaches
the ~520k the shared-arena design predicted -- **the allocator is not the
limit, the corridor is**. It OOMed because both phases' pools were
resident, which is exactly the duplication the arena removes. So the
arena's projected payoff is now measured rather than modelled.

Shipped: `RANK_MIB=20800,12650,13000`, `SGLANG_UNEVEN_TOKEN_VECTOR=28,26,20`.

| quantity | previous | shipped |
|---|---|---|
| TP pool | 318176 | **340474** (+7.0 %) |
| PP pool | 278104 | 278104 |
| corridor min under decode load | 2318/2187/2068 | 1986/2197/1672 |
| corridor min under 32768-tok prefill | not isolated | **1430/1633/1118** |

Corridor verified against the WORST load, which is the 32768-token
prefill rung, not decode: the prefill activation transient costs ~600 MiB
per card while decode costs ~70. An earlier candidate passed a
decode-only sample at 1986/2197/1672 and still breached at 716/1097/856
once the prefill ladder ran. A corridor number without the prefill rung
in it is not a corridor number.

Regression gate against the previous shipped config (A-vs-A, same script):

| point | previous | shipped | delta |
|---|---|---|---|
| decode narrative | 79.02 | 78.62 | -0.5 % (CV 2.3 %) |
| decode code | 103.21 | 101.31 | -1.8 % (CV 0.7 %) |
| prefill 2048 / 8192 / 32768 tok/s | 4227.8 / 7183.8 / 6712.8 | 4227.1 / 7184.7 / 6747.9 | within noise |
| flip pp_to_tp per rank | 955/1203/1678 ms | 958/1206/1681 ms | unchanged |

Pool growth cost nothing measurable: prefill is identical, decode is
inside its own CV with a slight negative trend worth re-checking if the
pool grows again.

## 6d. Cross-phase backing swap: what is built, what is not

Built and committed (`93ee1e2310`, `fa6ceab7b3`), 271/271:

- **Reads before writes at the flip.** The local leg read and wrote per
  layer in one loop, justified by "source and destination are different
  pools" -- the premise sharing removes. Hoisted, with an aliasing
  falsifier that builds both layouts as overlapping views into one arena
  and checks byte-identity against the disjoint reference. Proven red on
  the old loop.
- **Pool primitives.** `swappable_backing` allocates on a VA reservation
  and backs it fully at construction; `release_backing()` /
  `restore_backing()` unmap and remap physical pages behind UNCHANGED
  addresses. Sizing is untouched, so the #592 resize-translation gap does
  not apply -- the span is always the pool's own row count.
- **The seam.** `pre_write_fns` fires between the last source read and
  the first destination write: the only instant where backing may move.
  Pinned per rank (the ranks run concurrently, so a global event order
  proves nothing). Empty by default -- no behaviour change.

NOT yet wired, and each is a real step:

1. Construct both stacks' pools with `swappable_backing=True` and forward
   the two calls through `HybridLinearKVPool`.
2. Register the swap at the seam.
3. **Settle the allocator / tree-cache binding first.** `scheduler.tree_cache`
   and the allocator are bound to the PP stack's pools for process life,
   and the cutover does not rebind them (it swaps ps, groups, the batch
   result processor and the model worker; `scheduler.tp_worker` stays the
   PP stack). The TP forward uses the TP stack's pool, so the PP pool
   looks idle during decode -- but "looks idle" is not evidence, and
   releasing pages under a live reader corrupts KV silently. This needs
   proof before it is armed on production.
4. Re-budget: with exclusive backing, each pool may size against the FULL
   per-rank budget. Today's per-stack "double-spend" (the TP stack cannot
   see the PP pools already resident) stops being a bug and becomes the
   correct accounting.
5. Same treatment for the mamba pools (0.74+0.70 GB on rank 0). Note
   `GdnFlipMover` already packs every outgoing payload before writing, so
   it satisfies the reads-before-writes invariant with no change.

Expected on completion: the 496671-token measurement above, inside the
corridor instead of OOMing it.

## 6e. Correction: which pool is the serving capacity (commit c8be5d4d50)

**The TP pool is not the serving capacity, and an earlier claim in this
file that raising it from 318176 to 340474 was a "+7.0 %" gain is
WRONG.** The correction matters more than the number.

`Scheduler.build_kv_cache` builds ONE allocator, from the PP stack,
before the TP stack exists -- and the cutover never swaps it (it swaps
ps, groups, the batch-result processor and the model worker;
`scheduler.tp_worker` stays the PP stack). That is required, not an
oversight: the flip identifies a KV row by its GLOBAL slot id across both
layouts, so a single id space is what makes the transition expressible at
all.

So the id space -- and therefore the tokens a request can actually
occupy -- is the **PP stack's** capacity. On the shipped configuration
that is **278104**, unchanged by the TP-side work. The TP stack's 340474
is headroom above the id space, not usable capacity.

### The invariant nobody was checking

Every id the allocator can issue must be addressable in BOTH layouts, so:

    TP stack capacity  >=  PP stack capacity (the id space)

The TP stack derives its capacity independently, from its own budget and
token vector, so it can come out smaller. Then the first decode touching
a high id writes past the end of the TP KV pool -- and lands in
store_kvcache's own guard,
`SGL_DEVICE_ASSERT(index >= 0 && index < size_limit)`
(`jit_kernel/csrc/elementwise/kvcache.cuh:112`), a device-side assert
that kills all three ranks with an async CUDA error whose traceback
points at whatever host call synchronised next.

**That is the crash this rig hit mid-benchmark** at PP/allocator C =
46422 against TP C = 27200 (section 5, defect 6). It was recorded there
as "KV slot OOB under pool exhaustion"; the pool was not exhausted, the
TP pool was simply smaller than the id space. `c8be5d4d50` refuses that
shape at boot, naming both capacities and the knob that raises the TP
side.

### What this means for the capacity work

Raising the TP pool alone buys nothing. The lever is the **PP** pool,
and it is held down by the same double-residency: the PP pool is sized
first, against the full budget, but the budget must then still fit the TP
stack afterwards -- which is why 29000 MiB/rank produced a PP pool of
802904 and then OOMed building the TP stack.

Boot-time exclusive backing is therefore the piece that pays:

1. PP pool constructed and backed.
2. Release PP backing before the TP stack allocates its pools and
   captures its decode graphs.
3. Release TP backing, restore PP backing (the boot phase is PP).

Peak becomes max(PP, TP) instead of PP + TP at boot as well as at
runtime, and the per-rank budget can then rise into the ~500k-800k class
the 496671 measurement showed is reachable. The flip-seam swap
(fa6ceab7b3) is the runtime half of the same mechanism; the boot half is
what unlocks the sizing.

### Shipped state (commit c8be5d4d50, re-verified)

| quantity | value |
|---|---|
| serving capacity (allocator id space = PP pool) | **278104** |
| TP stack capacity (must be >= id space) | 340474 |
| corridor min, 100 ms, prefill rung + flip + decode | **1352 / 1599 / 1040** (floor 1024, HELD) |
| prefill 2048 / 8192 / 32768 | 4236.2 / 7175.7 / 6718.6 tok/s |
| decode narrative / code | 77.50 / 103.38 tok/s |
| flip pp_to_tp per rank | 950 / 1196 / 1674 ms |

A-vs-A against the previous shipped numbers: prefill within noise
(4227.8 / 7183.8 / 6712.8), decode narrative -1.9 % at CV 2.8 % and code
+0.2 % at CV 1.1 %, flip timings unchanged. No regression from the guard.

## 6f. Exclusive KV backing, shipped (commit 89572e996d)

The double residency is gone. Exactly one phase layout holds physical KV
pages at a time, on a fixed VA reservation so no address a captured graph
baked in ever moves:

```
boot   back PP -> RELEASE PP -> build TP pools + capture TP decode graphs
       -> release TP -> restore PP (the boot phase)
flip   at the read/write seam: release source, restore destination
```

Source-then-destination at the seam is deliberate. The reverse holds both
layouts for the width of the swap, and a corridor minimum is CONTINUOUS --
a few milliseconds counts. Releasing first is safe because every row the
transition owes is already in the payloads (93ee1e2310).

Measured page movement per rank: PP release hands back 4320 MiB (5090) /
2160 MiB (3080); TP release 5888 / 4480 MiB.

### What the falsifiers caught

**flush_cache killed every rank with cudaErrorIllegalAddress in the TP
phase.** `_flush_zero_kv_buffers` zeroed the scheduler's pool -- and the
scheduler's pool is the PP stack's, released while TP serves. This is the
answer to "does anything touch the PP pool during TP decode": yes, and it
was found by arming the release and letting it fail loudly rather than by
reasoning about call graphs. Fixed by zeroing every layout that HOLDS
pages and skipping the unbacked one, which also closes the converse gap
(zeroing only what the allocator reaches would leave the active TP layout
un-zeroed and quietly break the bit-for-bit-after-flush property).

**The TP stack's pool came up unswappable.**
`derive_tp_stack_server_args` deliberately clears `enable_phase_flip` on
the TP copy, so keying the flag on it caught only the PP side.

**256-token byte-identity is not a valid instrument on this rig.** Three
draws in ONE phase with NO flip give three different hashes -- the
documented upstream GDN prefill nondeterminism. The address-stability
check therefore runs inside the reproducible regime (short deterministic
answers), where output is byte-identical across a full flip round trip
with `cuda graph: True`.

### Capping the TP pool

The TP layout can only ever address ids the scheduler's allocator issues,
and that allocator is the PP stack's. Left uncapped the TP pool sizes
itself to its own budget -- measured **788026** against an id space of
367704 -- and hoards the VRAM the PP pool needs. `--max-total-tokens
500000` costs nothing (the surplus was unaddressable) and moved TP-phase
free memory from 663/2676/1117 to 3231/5528/3109 MiB. That is what let
the id space grow.

### Shipped

`RANK_MIB=22200,14700,14700`, `SGLANG_UNEVEN_TOKEN_VECTOR=28,26,20`,
`--max-total-tokens 500000` (now the script defaults).

| quantity | before (59540e846e) | shipped |
|---|---|---|
| **serving capacity (id space)** | 278104 | **367704 (+32.2 %)** |
| TP capacity (guard: >= id space) | 340474 | 500000 (capped; 788026 uncapped) |
| corridor min, prefill rung + 2 flips + decode | 1352/1599/1040 | **2198/4033/2292** |
| PP-phase free (idle) | ~2000 | 4931/4598/3723 |
| TP-phase free (idle) | -- | 2359/4476/2429 |

Regression gate, A-vs-A against 59540e846e:

| point | before | shipped |
|---|---|---|
| prefill 2048 / 8192 / 32768 tok/s | 4236.2 / 7175.7 / 6718.6 | 4225.7 / 7145.9 / 6715.4 |
| decode narrative | 77.50 (CV 2.8 %) | 78.64 (CV 1.5 %) |
| decode code | 103.38 (CV 1.1 %) | 103.19 (CV 1.8 %) |
| flip pp_to_tp per rank | 950 / 1196 / 1674 ms | 913 / 1138 / 1631 ms |

+32 % capacity at no cost: prefill within noise, decode within CV, flips
slightly faster.

### The honest ceiling, and what binds it now

**Not** the corridor -- 2198/4033/2292 MiB of free memory is 1.2-3.0 GiB
of unused slack above the floor. The binder is the per-rank
PHYSICAL-availability check in `_profile_available_bytes`, which refuses a
budget larger than what is free to that rank AT PP SIZING TIME:

```
rank 0 (5090):  holds  8.68 GiB + 13.26 free = 21.94 GiB ceiling
rank 1 (3080a): holds  4.87 GiB +  9.69 free = 14.56 GiB ceiling
rank 2 (3080b): holds  6.04 GiB +  8.51 free = 14.55 GiB ceiling
```

The shipped budgets sit just under those. So 367704 rather than the
~496k class, and the next lever is that ceiling -- the PP stack is being
sized against memory the TP stack has not yet claimed and, under
exclusive backing, never will hold at the same time. Whether that check
can safely account for the swap is the follow-up; it is a different
question from the one this commit answered.

## 6g. The sizing lever, examined and NOT taken (final)

The next lever after exclusive backing looked like
`_assert_budget_physically_available`: make it aware that the two phases
never hold pages together, so it stops sizing against their sum. It does
not work, because it was never summing them.

`Scheduler.init_model_worker` sizes the PP pool at `scheduler.py:1232`
(`init_memory_pools`) and builds the flip's TP stack only afterwards at
`:1249`. So when the check runs there is **no TP pool resident to
discount**. The check computes

    reachable = used_by_me + device_free / ranks_on_gpu

and on the 5090 that reads `8.68 GiB held + 13.26 GiB free = 21.94 GiB`
of a 32.6 GiB card -- roughly 9.4 GiB accounted as "held outside this
process" on a card carrying exactly one rank.

> **CORRECTION, 2026-08-09 (successor 14). The paragraph that used to
> follow blamed "the deferred #652 residual (the 5090's CUDA context
> seeing only 19.58 GiB total)". THAT IS FALSIFIED BY MEASUREMENT.**
>
> `scripts/probe_652_device_total.py`, bare process, no serving:
>
> ```
> cuda:0  RTX 5090  mem_get_info total = 32088 MiB  NVML total = 32607  shortfall +519
> cuda:1  RTX 3080  mem_get_info total = 20054 MiB  NVML total = 20480  shortfall +426
> cuda:2  RTX 3080  mem_get_info total = 20054 MiB  NVML total = 20480  shortfall +426
> ```
>
> The 5090's CUDA context sees **32088 MiB**. The only gap is the
> ~519/426 MiB driver carve-out the corridor rule already documents.
> There is no 13 GiB driver wall, so there is no #652 sizing residual to
> wait on.
>
> Where 19.58 GiB actually comes from: it is the **3080s'** `reachable`
> value, which sits two lines away in the same `BUDGET-REACH[nvml]` log
> family. The numbers were crossed when this section was written.
>
> What binds the pool instead, measured on both boots today, is that
> `BUDGET-REACH` reports **shortfall 0.00 GiB on every rank** -- the
> physical check is not binding at all. The binder is the hand-set
> `RANK_MIB` in `scripts/route_a_631_prod_boot.sh` (22700/11920/11970),
> which sits ~9.1/7.9/7.8 GiB below the measured reach and was chosen from
> corridor sampling. Raising it is an empirical boot-and-sample question,
> not a driver question.
>
> See HANDOFF_657.md sections 4 and 5, including the measured spill ladder
> (draft weights 1.86-2.01 GB/rank and draft graphs ~0.55 GB/rank are
> resident in BOTH phases while PP has no draft worker at all).

## 6h. Final acceptance (commit bc3016595d, production)

| quantity | value |
|---|---|
| **serving capacity (id space)** | **367704** |
| TP capacity | 500000 (capped; 788026 uncapped) |
| corridor min, 100 ms, prefill rung + flip + decode bench | **2286 / 4001 / 2380** (floor 1024, HELD) |
| prefill 2048 / 8192 / 32768 | 4236.8 / 7245.5 / 6842.6 tok/s |
| decode narrative / code | 76.67 (CV 0.9 %) / 101.99 (CV 2.5 %) |
| flip pp_to_tp per rank | 997 / 1246 / 1720 ms |
| probe (temperature 0) | 217 |

A-vs-A against the previous shipped state: prefill +0.0 / +1.0 / +1.9 %,
decode -2.5 % narrative and -1.2 % code (both within the run-to-run band
these points have shown all session), flips unchanged. Dashboard up on
8780, crash watchdog still decommissioned.

## 7. Reproduce

```
# production boot (leaves the server in the PP prefill phase)
HICACHE=0 RANK_MIB=20800,12500,13600 SGLANG_UNEVEN_TOKEN_VECTOR=13,11,8 \
  bash scripts/route_a_631_prod_boot.sh

# prefill, PP phase
python3 scripts/route_a_631_prefill_ladder.py --port 30030

# flip to decode
curl "http://127.0.0.1:30030/flush_cache?timeout=60"
curl -X POST http://127.0.0.1:30030/phase_flip \
     -H 'Content-Type: application/json' -d '{"direction":"pp_to_tp"}'

# decode, TP phase
URL=http://127.0.0.1:30030 MODEL=Qwen3.6-27B RUNS=5 WARMUPS=3 \
  bash /spinning/llm_stuff/club-3090/scripts/bench.sh

# dashboard
python3 -m sglang.planner --serve --port 8780
```

Test family: `bash scripts/run_631_flip_family.sh` — 266/266.

---

## 8. Flipping UNDER LOAD (#631 J.3 and after, 2026-08-09)

Until this section every flip in the feature's history committed with an
EMPTY pipeline, so "the flip works" was a statement about the flip and
never about the requests. Three defects sat between that and a flip under
load, and each was found by a leg that could not commit rather than by
reading code.

**J.3 -- the cutover dropped the resident decode set.** Step 6 of the
cutover calls `init_pp_loop_state()`, which rebinds `running_mbs` to
fresh empty batches -- and under `event_loop_pp` that array IS the
resident decode set. The stranded KV page and the stranded mamba lock
(`x_lru.full_lock_ref=1` -> SIGQUIT) were two symptoms of that one
omission. The carry now lives inside `init_pp_loop_state` itself, because
`event_loop_pp` calls it again at its own entry and would otherwise wipe
a carry installed at the cutover. See `phase_flip_resident_carry.py`.

**L -- `tp_to_pp` could not reach quiescence under load.** Under
`event_loop_normal` the result is processed in the SAME iteration and
`last_batch = batch` is set afterwards, so a non-empty `last_batch` at
the hook means "requests are resident", not "work is in flight". The
predicate refused on it, so the return leg parked and abandoned for as
long as anything was decoding. Same category error as the
`_pp_microbatches_drained` one that blocked every automatic flip before
it. Quiescence now asks the narrower question the carry needs: is every
live request reachable through the handle the carry harvests?

**M -- the PP chain's ring was read off the live `ps`.** The cutover
rewrites `ps` per phase and the TP phase gets `pp_rank=0, pp_size=1`, so
`(pp_rank - 1) % pp_size` made UPSTREAM == SELF on every rank. The
flip-commit hygiene check then compared a rank's own dict SEND counter
against its own dict CONSUME counter -- two different wires -- and rank 0,
the first PP stage, sends proxy dicts and consumes none. Measured: 8889
withheld rounds, "tensor-dict wire has 24 unconsumed message(s) from rank
0" (itself), `tp_to_pp` abandoned for want of a quorum it could not form.
The counters are built once from the PP topology at boot and are now the
ring's one authority.

### The survival oracle

`scripts/route_a_631_survival_oracle.py` is the standing harness: a
determined counting probe (chat, thinking off) decodes while the flip
happens underneath it.

```
# reference: no flip, records what the build answers when nothing moves
python3 scripts/route_a_631_survival_oracle.py --port 30030 --reference

# one leg, and both legs under ONE request
python3 scripts/route_a_631_survival_oracle.py --port 30030 \
    --direction pp_to_tp --flip-after 15 --limit 700
python3 scripts/route_a_631_survival_oracle.py --port 30030 --round-trip
```

Two properties of it are worth keeping, because both were bought with a
misleading result:

* **It reads COMMIT evidence from the serving log, not from the HTTP
  response.** `/phase_flip` returns 200 for ARMED; a leg that parks and
  abandons returns 200 too. A refused leg once read as a green round
  trip.
* **The verdict is anchored to the flip point**, not to total length: it
  counts how many integers are still CORRECT after each cutover. The
  first probe was a raw completion, and the model editorialised about how
  far to count -- the no-flip control drifted EARLIER (32 numbers) than
  the flip run (43), which is what proved the drift was the model and not
  the cutover.

### `SPEC=off`

`SPEC=off bash scripts/route_a_631_prod_boot.sh` boots the TP decode
phase without the NEXTN draft worker. It is not a convenience knob: a
request that prefills in the PP phase has no draft KV, because the PP
phase carries no draft worker at all, so a carried request entering a
speculating TP phase raises a second question on top of the carry.
SPEC=off answers the carry alone.

---

## The LONG-CONTEXT configuration (2026-08-09) — one session past 262144

A second boot recipe, alongside the standard one. It trades ~14 % of the
mixed-traffic KV pool arrangement for a per-session context ceiling above
the model's native 262144, and it is the configuration the bs=1
long-context leg was proven on.

```
MODEL=/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8-yarn1.5 \
CTX=393216 POLICY=auto SPEC=on PHASE_POLICY_TP_TOK_S=1681.0 \
bash scripts/route_a_631_prod_boot.sh --gdn-resident-state-slots 10
```

Two flags, and NEITHER is optional:

* the **overlay model directory** carries the YaRN rope block
  (`scripts/dev/543_yarn/make_yarn_overlay.py --factor 1.5`). The flag
  routes `--json-model-override-args` and `--decrypted-config-file` do not
  reach `text_config.rope_parameters` on this tree (#543).
* `--gdn-resident-state-slots 10` is what makes the raised ceiling
  REACHABLE. Raising `--context-length` alone SHRINKS the PP-phase pool
  (263768 -> 253528 tokens) and drops `max_req_input_len` BELOW 262144, so
  the session gets shorter, not longer. The cap moves 0.37 GB (rank 0) /
  0.18 GB (each 3080) out of the GDN state pool and into KV:

      PP-phase pool        253528 -> 277468 tokens
      max_req_input_len    253522 -> 277462
      max_running_requests      4 -> 4       (pre-cap admission ceiling)

### Measured on this configuration

    one session, prompt_tokens 276283 (the server's count)
      three needles at 3 % / 55 % / 95 % depth, per-run random codes
      all three returned verbatim, including the 95 % one, i.e. past the
      native ceiling. Second question on the same prefix: 6.37 s off the
      prefix cache.
    prefill of that session: ~1000-1260 tok/s at 2048-token chunks
      (against 4480 tok/s for a 32768-token prompt from rest -- the cost
      is the context, not the configuration)

    unmanned acceptance, 564 s, POLICY=auto, no client flip call:
      27 PHASE-FLIP DONE tp + 27 DONE pp (both directions, 3 ranks)
      165 accept-len lines (2.92-2.96), 165 graph decode passes
      109/109 requests ok, 0 aborted
      CORRIDOR HELD: min free 1250 / 3336 / 1388 MiB, 0 breaches

    evidence: /spinning/evidence-631/unmanned_acceptance_20260809T154146Z/

### Read this before quoting a pool number

`max_total_num_tokens` as printed by `/get_server_info` and by the boot
banner is the **PP-phase** pool. The larger figure in the uneven-DCP sizing
lines (459392 on the standard config) is the **TP-phase** pool. A request is
admitted against the PP one. Three shifts of handoff quoted the TP number as
"the pool" and concluded that ~197k tokens were unreachable by a single
session; what was actually true is that the PP pool sat about 1600 tokens
above the context ceiling.

### Not yet the default, and why

A flip attempted minutes after such a session takes the instance down in
`phase_flip_runtime._live` (`torch.cat` of a 1-D and a 3-D part). See
HANDOFF_656 v14 section 4 for the reproduction and for why reshaping is the
wrong fix.

### Long-context configuration: the full acceptance run (2026-08-09, updated)

The recipe above is now the configuration serving runs on, and its
acceptance evidence is one unattended log carrying every axis at once:
`/spinning/evidence-631/unmanned_acceptance_20260809T160920Z/`.

    33 PHASE-FLIP DONE tp + 36 DONE pp    both directions, policy-driven
    126 armed, 57 abandoned (38 for staging room while the pool was full)
    150 accept-len lines, 150 CUDA-graph decode passes
    WIRE accept-len: spec_accept_length 9.52, rate 0.36, 7 requests
    bs=1 session 276255 tokens, three depth needles verbatim
    80/80 requests ok, 0 aborted, health 200 at the end
    CORRIDOR HELD: 1210 / 3310 / 1386 MiB min free, 0 breaches

Two measurement notes worth keeping:

* **accept-len is not on the OpenAI route.** `/v1/chat/completions` returns
  an empty `meta_info`; the counters are on native `/generate`, and only for
  a request that actually speculated -- which here means one that verified
  in TP. A short request at rest runs entirely in PP and returns nothing,
  which is correct and looks exactly like a broken wire. Use
  `scripts/route_a_631_wire_accept_probe.py`, which issues concurrent decode
  work to move the policy into TP first.
* **A flip at full pool is refused, not attempted.** 38 of the 57 abandons
  in this run were the staging guard: at 0.99+ occupancy the exchange wants
  4887-6984 MiB of staging and cannot have it without eating the 1024 MiB
  reserve. Serving continues in the current layout. This is why the long
  session no longer takes the instance down.

The `_live()` crash that kept this configuration off the default in the
previous entry is fixed (`ce61812870`); the reproduction now runs as PHASE 3
of the unmanned acceptance script.

---

## A-vs-A REGRESSION GATE: `7ed8abfddc` vs `9a929352c9` (2026-08-09) — PASS

The ship gate. It asks one question and only that one: **does carrying the
phase-flip build cost anything on the path a user gets when they do not ask
for the flip?** Evidence:
`/spinning/evidence-631/ava_gate_20260809/`.

### How it was run, and why that shape

* **Both boots run WITHOUT `--enable-phase-flip`.** `route_a_631_prod_boot.sh`
  always passes it, so the gate has its own boot script,
  `scripts/route_a_631_ava_boot.sh`, parameterised by `WT`. Every other knob
  is identical; the only difference between the two invocations is the tree.
* **Speculation is OFF, and that is not a choice.** `check_server_args`
  refuses speculation under pipeline parallelism unless the flip is enabled,
  because the draft worker only exists on the flip's TP stack. The non-flip
  default path at `pp_size 3` cannot speculate in EITHER tree.
* **The two boots sized identically**: `max_total_num_tokens=302072` and per
  rank `available_gpu_mem` 9.94 / 9.30 / 6.92 GB, the same numbers on both
  commits. The flip build does not move the non-flip memory plan.
* **Per-rank ms/round, split compute vs wait.** The usual
  `Prefill rank batch ... gpu-ms (compute, wait)` line is NOT available here:
  `_install_rank_prefill_timer` returns early for `pp_size != 1`, identically
  in both trees. What every PP rank does emit under
  `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` is `fwd occupancy` — GPU-busy time
  over the wall window, i.e. the compute fraction of that rank's round.
  Harvested by `scripts/route_a_631_ava_rounds.py`.

### THE FINDING THAT ALMOST BECAME A FALSE REGRESSION

The baseline's first pass looked like the flip build was **8 % faster than it
should be**, i.e. a regression of the same size in the flip build. It was not.
The baseline pass had been started on cold cards and decayed monotonically as
the 3080s reached their thermal ceiling:

    baseline, from cold:   4106 -> 4011 -> 3953 -> 3887 -> 3834 -> 3809 tok/s
    same boot, after soak: 3756 -> 3762 -> 3770 -> 3773 -> 3780 -> 3776 tok/s

The flip boot had, by accident of sequencing, soaked before its measured
block and so was already at steady state. **The A/A design is what caught
this**: the cold pass's own two blocks disagree by -4.47 %, seventy times the
soaked pass's +0.38 %. A cold pass is therefore INADMISSIBLE by rule
(`STEADY_STATE_TOLERANCE_PCT = 1.0` in `route_a_631_ava_verdict.py`), in
either direction — whichever build happens to be measured cold looks worse,
and quoting either number would have been a fabricated finding. The cold pass
is kept on disk as `base.pass1_cold.json` precisely so nobody re-derives it.

### The numbers

Content axis fixed (one 48400-token prompt, cache flushed per rep; one greedy
512-token continuation), warmup discarded, 2 x 3 back-to-back reps per rung,
every rep above the 10 s floor (12.8 s prefill, 16.7 s decode).

| rung | baseline `9a929352c9` | flip `7ed8abfddc` | delta | same-boot floor | verdict |
|---|---|---|---|---|---|
| prefill tok/s | 3769.48 | 3769.40 | **-0.002 %** | 0.655 % | inside |
| decode tok/s | 30.602 | 30.624 | **+0.071 %** | 0.279 % | inside |

Per-rank ms/round, compute vs wait — the two commits are the same machine:

| rung | rank | baseline ms/round (compute/wait) | flip ms/round (compute/wait) |
|---|---|---|---|
| prefill | PP0 (5090) | 624.1 (221.8 / 402.3) | 624.1 (221.1 / 403.0) |
| prefill | PP1 (3080a) | 624.1 (607.3 / 16.8) | 624.1 (606.5 / 17.6) |
| prefill | PP2 (3080b) | 619.4 (585.3 / 34.1) | 614.8 (575.7 / 39.2) |
| decode | PP0 | 34.1 (9.3 / 24.8) | 34.1 (9.3 / 24.8) |
| decode | PP1 | 34.1 (10.0 / 24.1) | 34.1 (10.0 / 24.1) |
| decode | PP2 | 34.1 (13.9 / 20.2) | 34.1 (13.9 / 20.2) |

Read the wait SPREAD, not its level: the 5090 sits at 35 % occupancy and
waits 403 ms of every 624 ms prefill round for the 3080s, which are at 94-97 %.
The pacemaker is the 3080 pair, on both commits equally.

One more datum that is not a throughput number: the greedy 512-token
continuation returned the **same output hash (`9db721974590`) on both
commits**, every rep of both boots. The non-flip path is not merely as fast
as before, it is bit-stable across the 91 commits.

**VERDICT: PASS.** Both rungs admissible, both deltas inside the same-boot
floor. The flip build does not regress the non-flip default path.

## SEAM PEAK: what one PP<->TP cutover costs in VRAM (2026-08-09, successor 16)

The headroom conversation in this task had been conducted entirely in
units of the 1024 MiB corridor floor. That floor is a STEADY-STATE
budget, and a cutover is not steady state: it stages KV backing, packs
GDN slots and checksums the payload, transiently. This is the first
measurement of that transient.

Instrument: `scripts/seam_peak_measure.sh` — NVML free at 100 ms,
baseline = median of the pre-flip window, peak = baseline - trough. NVML
rather than torch's accounting, because the driver-level number is what
refuses a `cuMemCreate`.

### Point 1 — pool 253528, ctx 393216 (yarn1.5), boot 22:04:14Z

direction `pp_to_tp`, 110 samples over 11.0 s:

| card | baseline free | trough free | SEAM PEAK | trough above floor |
|---|---|---|---|---|
| gpu0 5090  | 5705 MiB | 2725 MiB | **2980 MiB** | 1701 MiB |
| gpu1 3080a | 6248 MiB | 4872 MiB | **1376 MiB** | 3848 MiB |
| gpu2 3080b | 4507 MiB | 2679 MiB | **1828 MiB** | 1655 MiB |

### What this already changes, before any second point

HANDOFF_658 §4e records ranks sitting ~530-610 MiB above the corridor
floor at runtime. Against a transient of 1.4-3.0 GiB that is not a thin
margin — it is a guaranteed death the next time a flip lands, which is
what three consecutive boots did (21:43Z, 21:51Z, 21:56Z).

The binding constraint on pool size is therefore

    pool_ceiling ~ card_total - corridor_floor(1024) - seam_peak

and NOT the corridor floor alone. Any pool sized against the floor is
sized to kill the next cutover. This puts the >600k full-KV goal in
NUMERIC tension with auto-flip rather than suspected tension.

Caveat carried honestly: one cutover, one direction, one pool size. It is
not a distribution, and the direction may not be symmetric.

### Point 2 — pool 126000, everything else IDENTICAL (boot 22:14:12Z)

Produced by `scripts/seam_scaling_reboot.py`, which replays the live
launch read back from `/proc` (85 env vars, 56 arguments) with exactly
one substitution: `--max-total-tokens 500000 -> 126000`. The per-card
budget stayed `22700,11920,11970`, ctx stayed 393216, weights and
corridor untouched. Pool went 253528 -> 126000 (x0.497).

| card | baseline free | trough free | SEAM PEAK | vs point 1 |
|---|---|---|---|---|
| gpu0 5090  | 6893 MiB | 6525 MiB | **368 MiB** | 2980 -> 368 (x0.12) |
| gpu1 3080a | 8586 MiB | 8586 MiB | **0 MiB**   | 1376 -> 0 |
| gpu2 3080b | 5677 MiB | 5629 MiB | **48 MiB**  | 1828 -> 48 (x0.026) |

**Direction confound checked and excluded.** Both windows straddle the
same PAIR of cutovers, verified from the logs rather than assumed:

* point 1, window 22:08:37-22:08:52 — epoch 3 `pp_to_tp` 22:08:41,
  epoch 4 `tp_to_pp` 22:08:43;
* point 2, window 22:16:56-22:17:11 — epoch 3 `pp_to_tp` 22:17:05,
  epoch 4 `tp_to_pp` 22:17:06.

Same directions, same instrument, same window length. (A passive 40 s
sample taken afterwards caught NO cutover at all: an idle instance stops
flipping, so the flips must be driven, not waited for.)

### ANSWER: the seam peak SCALES with the pool, steeply

Halving the pool cut the driver-visible cutover transient by 8x to 38x —
far more than proportionally. **The seam peak is not a fixed budget line
that a bigger pool can be sized around.**

The mechanism this implies, and it matters more than the ratio: the
instrument measures the DRIVER-visible transient, which is the right
observable because that is exactly what refuses a `cuMemCreate`. What the
seam must ask the DRIVER for is (staging size) minus (slack torch already
holds). A bigger pool grows the staging AND shrinks the slack, so the two
terms move against each other and the driver-visible peak grows much
faster than the pool. That is the same mechanism defect N died of: the
backing path called `empty_cache()`, handing back the very slack that
would have absorbed the next 128 MiB, and the driver then refused it.

**Consequence for full-KV.** A >600k pool cannot be reached by sizing
around this peak, because the peak grows as the pool does. It requires a
fundamentally cheaper seam — pre-reserved, zero-allocation staging so the
cutover never asks the driver for anything — not a bigger headroom
allowance. The pool-ceiling formula from point 1 stands, but with a
seam_peak term that is a function of the pool rather than a constant.

## CORRECTION (successor 17, 2026-08-09): the "seam peak" is the TP-phase PLATEAU, not a cutover transient

The headline of the previous shift -- "one pp_to_tp cutover costs 1.4-3.0
GiB per card" -- is a MISATTRIBUTION, and every conclusion drawn from it
(full-KV and auto-flip are in numeric tension; the seam must be made
zero-allocation before capacity work can proceed) has to be withdrawn.

### What was actually measured

The original sampling window straddled a there-and-back flip PAIR. The
previous holder note says so in its own words -- "both sampling windows
straddle the same pair, epoch 3 pp_to_tp + epoch 4 tp_to_pp" -- and that
is exactly the confound: what was read as a transient trough between two
baselines is the TP-phase plateau BETWEEN the two flips.

It reproduces on demand. Driving one `pp_to_tp` on the idle instance
looks like a clean 1.6 s transient:

    t=0.1s  5705 6248 4507     (PP)
    t=0.9s  2745 4872 2699     <- "the seam peak"
    t=1.7s  2745 4872 2699
    t=2.5s  5705 6248 4507     (recovered)

but the log shows the instance flipped BACK on its own 1.5 s later
(`PHASE-FLIP DONE tp_to_pp (epoch 4)`, 22:44:30Z): POLICY=auto returns an
idle instance to PP. The "recovery" is the second cutover, not the end of
a transient.

### The falsifier: hold the instance in TP with real work

Same instrument, but a 1200-token generation keeps the instance in the TP
layout for the whole decode (request wall time 7.93 s):

    t=0.1s  5705 6248 4507     PP, before
    t=1.1s  2743 4850 2695     TP  <-+
    t=3.1s  2739 4848 2693     TP    | FLAT for ~7 s, with no cutover
    t=8.1s  2739 4848 2693     TP  <-+  running anywhere in the window
    t=9.1s  5699 6222 4501     back in PP
    t=39s   5679 6222 4481     still PP, stable

A cutover lasts 1.0-1.7 ms-scale seconds (`PHASE-FLIP DONE ... in 961.8 /
1218.8 / 1711.9 ms`). A plateau that holds flat for 7 s and ends when the
REQUEST ends is a steady state, not a transient.

**So the number is real but it is a PHASE HOLD:** the TP layout keeps
~2960 / 1400 / 1810 MiB more VRAM resident than the PP layout does, for
as long as it is the active layout.

### Why the pool-scaling result also dissolves

Point 2 (pool 126000, "peak" 368/0/48 MiB) was produced by substituting
`--max-total-tokens 500000 -> 126000`. That flag caps the TP pool. It did
not shrink a transient super-linearly; it removed the TP/PP span
difference, so the two plateaus became the same height and the apparent
peak vanished. Nothing here is superlinear and nothing needs a
zero-allocation seam to be reachable.

### What actually binds capacity, with the numbers from this boot

    PP pool (= the id space = serving capacity)   253528 tokens
    TP pool (sizes itself to its own budget)      450290 tokens
    --max-total-tokens cap                        500000  -> NOT BINDING

Per section 6e the id space is the PP pool, so the TP pool's surplus of
~196762 tokens is UNADDRESSABLE by construction -- no request can ever
occupy it -- and it is exactly the hoarding that 6f capped with
`--max-total-tokens`. At this boot the cap was set above the TP pool's own
sizing, so it does nothing and the hoard is back.

**The lever is therefore the one 6f already used, and full-KV does not
need new machinery.** Capping the TP pool just above the id space
(honouring the 6e invariant `TP capacity >= PP capacity`, whose violation
is a device-side assert) releases the plateau difference during the TP
phase, and that is the headroom the PP pool -- the number that actually
sets serving capacity -- can grow into.

### What survives from the previous shift

* The measurement itself and its instrument: correct, reproducible, and
  the raw samples are honest. Only the ATTRIBUTION was wrong.
* Handle retention / a zero-allocation seam (shipped this shift, default
  OFF behind `SGLANG_FLIP_SEAM_CHUNK_MIB`) is still worth having: it
  removes the `cuMemCreate` OOM that can strike inside the cutover's
  no-return region, which is a measured crash (2026-08-09 12:47:45,
  rank 1). It is a CRASH fix, not a capacity fix, and must not be quoted
  as one.

### The general lesson, since this is the third number the chain inherited wrongly

A memory reading taken across a flip PAIR cannot distinguish a transient
from a phase hold. The separator is cheap and should be standard for any
future seam measurement: hold the target layout under load for
substantially longer than one cutover, and check whether the level tracks
the WORK or the CUTOVER.

### Confirmation on metal: capping the TP pool releases the plateau (successor 17)

One variable changed against the boot above: `MAX_TOTAL_TOKENS=260000`
(just above the PP id space of 253528, honouring the 6e invariant
`TP capacity >= PP capacity`). Same model, same CTX 393216, same
`RANK_MIB=22700,11920,11970`, same token vector. Measured with
`scripts/phase_plateau_measure.sh` (one 1200-token generation, 100 ms NVML
sampling, plateaus taken as the dwell-weighted modes rather than the
endpoints so a cutover sample cannot be mistaken for a level).

| card | TP-phase hold, cap 500000 | TP-phase hold, cap 260000 |
|---|---|---|
| 0 | 2960 MiB | **874 MiB** |
| 1 (5090) | 1400 MiB | **864 MiB** |
| 2 | 1810 MiB | **234 MiB** |

| card | corridor minimum (TP phase), cap 500000 | cap 260000 |
|---|---|---|
| 0 | 2739 MiB | **4931 MiB** |
| 1 (5090) | 4848 MiB | **6466 MiB** |
| 2 | 2693 MiB | **4373 MiB** |

PP-phase free also rose (5705/6248/4507 -> 5805/7330/4607). The PP pool is
unchanged at 253528, as expected: `--max-total-tokens` caps the TP pool
only, and the id space is the PP pool.

**~1.6-2.2 GiB per card of CONTINUOUS headroom, recovered by a flag that
was already in the tree and was simply set above the TP pool's own
sizing.** Against the 1024 MiB floor this leaves roughly 3.9 / 5.4 / 3.3
GiB per card to spend on the PP pool, which is the number that actually
sets serving capacity. That is the full-KV route, and it needed no new
machinery -- only the withdrawal of the transient reading that had made it
look impossible.

### Spending the headroom: serving capacity 253528 -> 300000 (successor 17)

`RANK_MIB=25700,13920,13970` (from 22700,11920,11970) with
`MAX_TOTAL_TOKENS=300000`. Boot 22:55:44Z, healthy in ~2.5 min.

**A correction to the model in the section above:** `--max-total-tokens`
is a GLOBAL cap, not a TP-only one. At 300000 it binds BOTH pools, so the
PP pool (the id space) rose to 300000 AND the TP pool sits at the same
number -- which means the unaddressable surplus is **zero by
construction**, while still satisfying 6e's `TP capacity >= PP capacity`
as an equality. That is a better configuration than "cap just above the id
space": there is nothing left to hoard.

| quantity | pool 253528 (cap 260000) | pool 300000 (cap 300000) |
|---|---|---|
| serving capacity (id space) | 253528 | **300000 (+18.3 %)** |
| corridor min, card 0 | 4931 MiB | 4483 MiB |
| corridor min, card 1 (5090) | 6466 MiB | 5688 MiB |
| corridor min, card 2 | 4373 MiB | 4051 MiB |
| TP-phase hold | 874 / 864 / 234 | 954 / 1152 / 188 |

So +46472 tokens of real capacity cost ~448 / 778 / 322 MiB of continuous
corridor. Against the 1024 MiB floor there is still ~3459 / 4664 / 3027
MiB per card unspent, and card 2 is the binding one.

**What that implies for the >600k full-KV goal**, stated as an
extrapolation and not a measurement: at ~322 MiB per 46472 tokens on the
binding card, the remaining 3027 MiB is worth roughly 400k further tokens.
The goal is very likely reachable by this route alone. It will stop
earlier than the arithmetic suggests -- `_profile_available_bytes` (bench
6f's "honest ceiling") binds the PP budget independently of the corridor
-- so the number to trust is the next measurement, not this paragraph.
