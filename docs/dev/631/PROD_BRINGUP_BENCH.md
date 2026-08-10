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

### The capacity ladder, measured (successor 17, 2026-08-09)

Three boots, same model / CTX 393216 / token vector, `MAX_TOTAL_TOKENS`
set equal to the target so it binds BOTH pools (zero unaddressable
surplus, 6e satisfied as an equality):

| RANK_MIB | cap | serving capacity | corridor min (TP phase) |
|---|---|---|---|
| 22700,11920,11970 | 500000 | 253528 | 2739 / 4848 / 2693 |
| 22700,11920,11970 | 260000 | 253528 | 4931 / 6466 / 4373 |
| 25700,13920,13970 | 300000 | **300000** | 4483 / 5688 / 4051 |
| 29200,15920,15970 | 360000 | **360000** | 3803 / 4700 / 3499 |

**+42.0 % serving capacity over the shipped baseline**, with the corridor
minimum still 2779 / 3676 / 2475 MiB above the 1024 MiB floor. Card 2 is
the binding one throughout.

Marginal cost, 300000 -> 360000: 680 / 988 / 552 MiB of continuous
corridor per 60000 tokens. At that rate card 2's remaining 2475 MiB is
worth roughly 269000 further tokens, i.e. a ~629000-token class -- which
would meet the >600k full-KV goal. Treat that as the next experiment, not
a result: `_profile_available_bytes` (6f's "honest ceiling") binds the PP
budget independently of the corridor and is expected to stop the ladder
before the corridor does.

Note what did NOT have to happen for any of this: no zero-allocation
staging, no new machinery, no change to the seam. The capacity was behind
a cap that had been set above the pools' own sizing.

### The ladder continued, and where it really stops

| RANK_MIB | cap | serving capacity | corridor min (TP phase) |
|---|---|---|---|
| 29200,15920,15970 | 360000 | 360000 | 3803 / 4700 / 3499 |
| 32200,17700,17750 | 520000 | **REFUSED AT BOOT** | -- |
| 31800,17400,17450 | 460000 | **460000 (+81.5 %)** | 2655 / 3034 / 2609 |

**Boot 4 found the honest ceiling exactly where 6f said it would**, and it
refused cleanly instead of dying:

    ValueError: The per-rank budget of 32200 MiB (31.45 GiB) for rank 0 on
    GPU 0 is not physically available: the rank holds 15.64 GiB and
    15.68 GiB of the device is free to it (31.34 GiB total)

That is `_profile_available_bytes`. The 5090's usable ceiling is ~31.3 GiB,
so rank 0's budget cannot exceed ~32000 MiB. Fail-fast worked: no crash, no
corrupted state, and the message names the numbers needed to pick the next
rung.

**THE BINDER HAS CHANGED, and this is the finding to carry forward.** At
`RANK_MIB=31800,17400,17450` the engine reports its own budget-derived
capacity as **1096606 tokens** -- more than twice what is configured -- so
`--max-total-tokens` is now the only thing holding the pool down, and the
BUDGET is no longer the constraint. What binds instead is the corridor:

    marginal cost, 360000 -> 460000:  1148 / 1666 / 890 MiB per 100k tokens
    headroom above the 1024 floor at 460000: 1631 / 2010 / 1585 MiB

Card 1 (the 5090) is now the binding card, and its 2010 MiB at 1666 MiB per
100k tokens is worth about +120000 more, i.e. a corridor-limited ceiling
around **580000** at this vector -- short of the >600k goal.

**The lever for the last stretch is the TOKEN VECTOR, and the engine
computes the recommendation itself:**

    Uneven DCP: restart with SGLANG_UNEVEN_TOKEN_VECTOR=31,17,16 to raise
    max_total_num_tokens from 1096606 to ~1396288 (per-rank profiled
    capacity [709932, 385302, 349078]; active vector [28, 26, 20] leaves
    ranks idle)

The active 28,26,20 over-weights rank 0, which is precisely the card whose
corridor now binds. Re-balancing toward 31,17,16 moves KV rows off the 5090
and should convert directly into corridor headroom on the binding card.
That is the next experiment, and it is a one-variable boot.

### Final rung this shift: 540000 tokens (+113 %), corridor held

One variable against the 460000 boot (cap 460000 -> 540000; `RANK_MIB`
unchanged at 31800,17400,17450), which the budget supports because the
engine's own capacity at these budgets is 1096606 tokens.

| quantity | 460000 | 540000 |
|---|---|---|
| serving capacity (id space) | 460000 | **540000 (+113.0 % on baseline)** |
| corridor min, card 0 | 2655 MiB | 1719 MiB |
| corridor min, card 1 (5090) | 3034 MiB | 1686 MiB |
| corridor min, card 2 | 2609 MiB | 1865 MiB |
| TP-phase hold | 1434 / 1792 / 282 | 1706 / 2080 / 448 |

Every card stays above the 1024 MiB floor, with 695 / 662 / 841 MiB of
margin. That is the direction the corridor rule asks for -- "never breach
AND as full as possible" -- and the capacity ladder for this shift is:

    253528 -> 300000 -> 360000 -> 460000 -> 540000    (+113.0 %)

**READ THIS BEFORE SHIPPING 540000.** The corridor figure above comes from
ONE 1200-token generation at bs=1. It is NOT a sustained-load measurement
and it is NOT the >=60-min bar. The design point in the user spec is bs=4,
where four concurrent requests and a fuller prefix cache will push the
minimum lower than a single stream does, and 662 MiB of margin on the
binding card is not much to give away. Treat 540000 as the measured
CEILING and 460000 (margin 1631/2010/1585) as the conservative fallback
until a sustained bs=4 run has held the corridor for an hour. The
instrument for that is a load generator, not `phase_plateau_measure.sh`.

---

## THE bs=4 CORRECTION (successor 18, 2026-08-09): 540000 is not a rung

The previous section closes by asking for the bs=4 run before 540000 is
believed. That run has now happened, and it does not confirm the rung — it
retires it.

Instrument: `scripts/route_a_631_bs4_rung.sh`, which drives the DESIGN
POINT (3 long generations plus one 12000-token prompt worker = the 4
concurrent requests `--max-running-requests 4` actually admits) while
sampling the NVML free column at 100 ms and judging the running minimum.

### The measurement

Same boot, same `RANK_MIB=31800,17400,17450`, same CTX 393216, cap 540000:

| card | bs=1 (the inherited number) | bs=4 (the design point) | breaches |
|---|---|---|---|
| nvml gpu0 (rank 1, 3080) | 1719 | **355** | 329 |
| nvml gpu1 (rank 0, 5090) | 1686 | **42**  | 252 |
| nvml gpu2 (rank 2, 3080) | 1865 | **661** | 465 |

602 samples, floor 1024, **all three cards breach**, the 5090 to 42 MiB.
The concurrency delta is ~1.3-1.6 GiB per card, which is the whole margin.

Two properties of this measurement are worth keeping:

* **The allocator does not give the peak back.** With the load stopped the
  5090 still read 42 MiB free. A post-run snapshot therefore cannot clear a
  configuration that a time series condemns, and "it looks fine now" is not
  evidence about the corridor.
* **bs=1 is not a weaker version of bs=4, it is a different measurement.**
  The +113 % ladder was climbed entirely on single-stream readings. Every
  rung above 460000 in that table is unproven at the design point, and the
  top rung is now falsified. A rung is only real when the load that
  produced it is the load the server is configured to admit.

### What holds instead: 460000, in the SHIP shape

Booted with strict purity and BOTH fairness windows on — the configuration
the user's spec requires, not a permissive one:

    PHASE_FLIP_PURITY=strict  PP_WINDOW_S=15  TP_DECODE_FLOOR_S=10
    POLICY=auto  RANK_MIB=31800,17400,17450  MAX_TOTAL_TOKENS=460000

    corridor minimum   1412 / 1459 / 1487 MiB      0 breaches
    flips              104 in the window, both directions
    deaths             0        request errors 0

### The fairness windows were never the defect

HANDOFF_658 §4e condemned these windows on three boots that died within
minutes, and concluded they are "NOT SHIPPABLE on this rig at these
values". Two of those three deaths were allocation failures at the seam
(`cuMemCreate` OOM; `torch.OutOfMemoryError` for 128 MiB with 106 MiB
free), taken on a rig that §657 measured as sitting 530-610 MiB above the
floor. The same windows, at a pool that leaves ~1450 MiB, produced 104
flips with zero deaths.

The windows raise the flip rate; the flip rate was never affordable
because the pool was oversized. **A policy knob was blamed for a sizing
defect.** The general form is worth naming, because this chain has now
made it three times in a row: a component measured in a configuration
that is independently broken will be convicted of the breakage. Restore
the configuration before judging the component.

(If a seam OOM does return, `SGLANG_FLIP_SEAM_CHUNK_MIB` — successor 17's
zero-allocation seam — is the targeted mitigation and is still default 0.)

### The 65-minute green run at 460000 (successor 18, 2026-08-10)

Boot `23:54:03Z` on commit `1ba907f1b5` (the self-merge fix), window
`23:54:16Z - 00:59:29Z`, harness `scripts/green_criterion_631.sh`.

    RANK_MIB 31800,17400,17450   MAX_TOTAL_TOKENS 460000   CTX 393216
    POLICY=auto  PHASE_FLIP_PURITY=strict
    PHASE_POLICY_PP_WINDOW_S=15   PHASE_POLICY_TP_DECODE_FLOOR_S=10

| axis | result |
|---|---|
| corridor, 28069 samples @100 ms | min free **1397 / 1354 / 1451** MiB, **0 breaches** |
| flips, unmanned | **1845** — 924 `pp_to_tp`, 921 `tp_to_pp` |
| prefill records | **1096 in PP, 0 in TP** |
| decode records | **1377 in TP, 0 in PP** |
| CUDA graphs | **99.4 %** of decode records `cuda graph: True` |
| speculation | mean accept len **2.90** |
| requests | **760 ok, 0 err** |
| scheduler exceptions | **0** |
| purity verdict | exit 0 — PURITY HELD, both layouts used |

All four axes the spec asks to hold SIMULTANEOUSLY — auto flip, CUDA
graphs, MTP speculation, and the largest corridor-legal KV — held together
for 65 minutes unmanned.

**The self-merge guard fired 494106 times in the window.** That is not
noise: it is the measure of how close this configuration was to the
crash, since every one of those calls would have doubled a batch before
`1ba907f1b5`.

#### What this run does NOT establish

The user's green criterion (spec item 7) requires the instance to serve
**real Qwen agent tasks through the router on 30099**. Every one of the
760 requests in this window was the synthetic soak driver on
`127.0.0.1 /v1/completions`; the window log records **no** agent traffic.
An attempt to supply it from the operator session by dispatching local
`qwen` agents failed to reach this server at all — the agents completed
their work correctly, but the serving log gained **zero** request lines,
and the whole log carries exactly one `/v1/chat/completions` since boot.
So the local-agent lane is not currently routed to this instance. That is
a router/harness question, not a serving defect, and it is the single
remaining gate between this run and green.

**Do not report this configuration as green.** Report it as: every
measurable serving axis passed for 65 minutes, and the agent-backend axis
is untested because no agent traffic arrived.

---

# SUCCESSOR 19

## 1. The agent-traffic lane: the router was never the defect

HANDOFF_661 §9 left the green criterion blocked on "real Qwen agent tasks
through the router", with the symptom that dispatched agents ran correctly
while the serving log gained zero request lines. The router was suspected.

**The router is fine, and so is the server.** A direct probe through
127.0.0.1:30099 with `model=Qwen3.6-27B` returned 200 and appeared in the
serving log within the same second:

    router  POST /v1/messages -> local (model=Qwen3.6-27B)
    serving [01:08:13] "POST /v1/messages HTTP/1.1" 200 OK

**The defect is in how the agent is dispatched.** The `qwen`,
`local-model` and `local-model-think` agent definitions carry
`model: Qwen3.6-27B` in their frontmatter, and the router routes on an
exact match of that string (`router.py:380`, `model in app[LOCAL_MODELS]`).
Passing an explicit `model:` argument to the Agent tool **overrides the
frontmatter**, so the request goes out as `claude-*` and the router sends
it upstream to Anthropic. Proven by a control pair, same task, same agent
type, one minute apart:

| dispatch | router verdict |
|---|---|
| `subagent_type: qwen`, no model argument | `-> local (model=Qwen3.6-27B)` x2 |
| `subagent_type: qwen`, `model: haiku` | `-> upstream (model=claude-haiku-4-5-…)` x2 |

The router's own counter confirms the lane had simply gone unused rather
than broken: 14492 local routes lifetime, last one at 21:59Z — two hours
*before* successor 18's run, i.e. no `Qwen3.6-27B` request was ever issued
during that window.

**The trap is that a standing instruction causes it.** The memory law
`agent-modellwahl` says "model IMMER explizit, sonst Fable-Erbe" — always
name the model explicitly. Applied to a `qwen` agent that rule silently
defeats the agent's entire purpose. **For local-model agents, pass no
`model` argument at all.**

## 2. The KV byte model, validated at two pool sizes

Per-rank KV bytes are governed by a single expression:

    per-rank KV bytes = T x 32 KiB x max(layer_share_i, token_share_i)

where `T` is the global `--max-total-tokens`, `layer_share_i` is that rank's
share of the 64 layers in the **PP** layout, and `token_share_i` is its
uneven-DCP token share in the **TP** layout. The `max` is there because both
layouts' KV is backed by one arena sized `max(PP, TP)`, not `PP + TP`.

32 KiB/token whole-model = 16 full-attention layers (`full_attention_interval
4`) x 4 KV heads x 256 head_dim x 2 (K and V) at fp8_e4m3.

Validated against two independent boots, exactly:

| | predicted | measured |
|---|---|---|
| T=540000 PP rank0 K | 4.12 GiB | 4.12 GiB |
| T=540000 PP rank1/2 K | 2.06 GiB | 2.06 GiB |
| T=540000 TP rank0 | 204324 tok / 3.12 GiB | 204344 tok / 3.12 GiB |
| T=460000 PP rank0 K | 3.51 GiB | 3.51 GiB |
| T=380000 PP rank0 K | 2.90 GiB | 2.90 GiB |

## 3. A 12.1 % capacity tax nobody had costed: layer share != token share

The ship config runs `pp_layer_ratio [32,16,16]` (shares 0.500 / 0.250 /
0.250) against token vector `28,26,20` (shares 0.378 / 0.351 / 0.270). The
per-rank cost is the **max** of the two, so the rig pays

    max = (0.500, 0.351, 0.270),  sum = 1.121

i.e. **112.1 % of one whole KV cache** to store one KV cache. Cross-checked
against the allocation: 460000 x 32 KiB x 1.121 = 15.74 GiB, and the three
measured arenas sum to 7.02 + 4.93 + 3.79 = 15.74 GiB exactly.

Aligning the two vectors drives the sum to 1.000 and returns that 12.1 %
**for free, at boot, with no new machinery** — it is a flag change, not a
feature. This is the one capacity lever in this chain that costs nothing to
try.

**But it is in direct tension with the PP compute balance.** `pp_layer_ratio`
sets both the PP stage's compute share and its KV share. HANDOFF_661 §8
records the 5090 drawing only ~250 of 400 W during PP prefill, which argues
for giving it *more* layers; capacity argues for giving it *fewer*, because
its layer share (0.500) exceeds its token share (0.378) and is therefore the
term that binds. **The 5090 cannot simultaneously be the compute-heavy PP
stage and the capacity-efficient one.** Naming that trade-off is the result;
picking a point on it is a measurement that has not been made.

## 4. >600k: the honest closure, with the full asset table

Successor 18 closed this negative on the draft assets alone. The user's spec
item 6 orders a different mechanic — spill everything cold of the *inactive
layout* at the phase change, including the PP weight shards. That mechanic
is **already implemented and already banked**:

`weights_arena.py:2-8` and `phase_flip_boot.py:477-547` — both layouts'
parameters live one at a time in ONE arena per rank sized
`max(pp_bytes, tp_bytes)`, and a flip rewrites its contents from the other
layout's **pinned host image** (`snapshot_and_free(..., pin=True)`). The
inactive layout's weights are already in system RAM, at a cost of 59.75 GiB
of pinned host memory already being paid.

Measured arena sizes (`TP stack built` log line):

| rank | PP layout | TP layout | arena = max | idle tail | tail freeable in |
|---|---|---|---|---|---|
| 0 (5090) | 14936 MiB | 13163 MiB | 14936 MiB | 1773 MiB | TP phase (binding) |
| 1 (3080) | 6690 MiB | 7924 MiB | 7924 MiB | 1234 MiB | PP phase (wrong side) |
| 2 (3080) | 9115 MiB | 7924 MiB | 9115 MiB | 1191 MiB | TP phase (binding) |

So the residue the spec's mechanic could still reclaim is the arena's **idle
tail**, not the shards themselves — and two of three ranks free it in the
binding phase, which is better polarity than the draft-asset rung.

**It still cannot fund >600k, for a reason that is structural rather than
arithmetic: the KV pool is sized ONCE, at boot, from a one-shot free-memory
reading, and every runtime grow path is closed.**

* `_profile_available_bytes` (`model_runner_kv_cache_mixin.py:608-700`) takes
  a single `mem_get_info` snapshot at pool-construction time; nothing
  re-measures on the serving path.
* `--max-total-tokens` is a `min()` cap only (`:4406-4427`) — it can lower
  the profiled capacity, never raise it.
* `post_capture_resize_kv_pool` is off by default, is shrink-only via
  `cap_tokens`, and is excluded for the TP stack anyway.
* `runtime_set_backing_tokens` is hard-bounded by the boot VA reservation.
* `phase_flip_boot.py:636-650` refuses boot unless `tp_capacity >=
  pp_capacity`, because one allocator serves both layouts for process life.

**Therefore no runtime spill of any asset can add KV capacity.** Freed bytes
become idle card slack. Chasing runtime spill for capacity is the wrong
question, and this is the third handoff to arrive there; the right question
is what pool the rig can BOOT with and still hold the corridor.

`phase_flip_spill.py` is written but **not wired** — `grep` for
`get_spill_ladder` finds only the module itself, and `phase_flip_spill_depth`
is not a `ServerArgs` field. Its docstring describes call sites in
`_cutover` step 7b that do not exist.

## 5. 460000 is not corridor-legal under real agent traffic

The number that matters, and it retires the inherited ship number.

Successor 18 established 460000 on a synthetic soak with **no agent
traffic** — and the user's green criterion *requires* agent traffic. Adding
it breaks the corridor:

    window 01:16:24Z, pool 460000, soak (2 decode streams + 12000-tok
    prefill / 25 s) PLUS real qwen agent traffic through router 30099
    1982 samples @ 100 ms
      gpu0 (rank1, 3080) min  591 MiB
      gpu1 (rank0, 5090) min  270 MiB
      gpu2 (rank2, 3080) min  621 MiB
    floor 1024 -> ALL THREE BREACH

Not contamination: `nvidia-smi` showed only the three scheduler PIDs on the
cards throughout.

Two properties worth carrying:

* **Free memory swings by GiB across a flip** (the KV backing
  release/restore leg), so a corridor minimum is phase-dependent and a
  reading is only meaningful with the phase it was taken in.
* **A flat corridor reading can mean an idle server.** The first 90 s of this
  window read a rock-steady 1393/1352/1447 — that was the soak spinning up,
  not the corridor passing. A corridor sample taken before load arrives
  proves nothing, and its flatness is the tell.

## 6. The capacity ladder under real agent traffic

The corridor minimum falls as the **agent context size** grows, because the
allocator's high-water mark follows the largest request shape it has seen.
This makes the ladder load-dependent in a way the synthetic soak never
exposed.

| pool | load | gpu0 | gpu1 (5090) | gpu2 | verdict |
|---|---|---|---|---|---|
| 460000 | soak only (s18) | 1397 | 1354 | 1451 | 0 breaches |
| 460000 | soak + agents | 591 | 270 | 621 | BREACH x3 |
| 340000 | soak + light agents | 1191 | 1166 | 1089 | 0 breaches, margin 65 MiB |
| 340000 | soak + heavy agents | 1029 | 886 | 869 | BREACH gpu1, gpu2 |
| 260000 | soak + heavy agents | see green run | | | |

**A plateau is only a plateau for the workload mix in flight.** At 340000
the corridor read *identically* 1191/1166/1089 for two consecutive
2-minute buckets — a textbook steady state — and then breached within two
minutes of two large-context agents joining. The allocator had simply
finished growing for the shapes it had been shown. **Stress the worst
shapes early**: a breach discovered at minute 40 costs the entire window,
and deliberately provoking it at minute 10 cost ten.

**Change the pool, not the budget.** Lowering `RANK_MIB` from
31800/17400/17450 to 30500/16550/16500 made the 5090 *worse* — 852 MiB free
at **idle**, already under the floor — because the KV pool is a hard
requirement while `RANK_MIB` is advisory: a too-small budget against a
too-large pool simply overshoots. The budget stays at the proven values and
`--max-total-tokens` is the lever.

**A structural oddity worth a successor's attention:** `RANK_MIB=31800` on a
32607 MiB card leaves 807 MiB, which is **below the 1024 MiB floor by
construction**. Every configuration that held did so only because the engine
did not consume its whole budget. The budget and the corridor law have never
been reconciled with each other.

---

## SUCCESSOR 20 / 1 — DEFECT R: the resident-carry leak, and it was one indentation

The blocker that ended successors 18's and 19's windows is closed. It was
not in the phase-flip machinery at all; it was in the PP event loop's slot
bookkeeping, and the phase-flip feature merely created the conditions that
made it reachable.

### The defect

`last_mbs[slot]` must name *the batch that slot ran in its previous
iteration*. Its assignment in `_event_loop_pp_body` sat INSIDE
`if self.mbs[next_mb_id] is not None:`, sharing a block with the D2H sync
and `_pp_process_batch_result`. That quietly redefined the name as *the
last non-empty batch this slot EVER ran*.

The two readings differ only when a slot runs nothing while requests are
still resident in it — and **strict purity creates exactly that state on
purpose**: `get_next_batch_to_run` returns `batch_to_run = None` for a
resident decode batch in the PP layout (`phase_decode_blocked_here` ->
`ret = None`), and that None is the intended signal that the PP phase has
no work.

The stale entry is an EXTEND batch, so every later visit to the slot takes
the `is_extend()` branch and reaches `running_batch.merge_batch(last_batch)`,
which extends `reqs` **in place**. Once per cycle, forever.

### Why both existing defences were blind, and neither was wrong

* the self-merge guard compares `last_batch is running_batch` — the stale
  entry is a **distinct object**;
* `harvest_resident_batches` dedupes by `id(batch)` — the duplication is
  **inside one batch's `reqs`**.

A distinct object holding already-resident Reqs is the one shape that
defeats both at once. Entry K of `phase_flip_presence` predicted precisely
this for a non-idempotent merge.

### The correction that mattered most

**`claims 5` was never the onset.** 5 is merely the first value above
`max_running_requests=4` that the guard is *able to report*; the leak runs
silently through 1, 2, 3, 4 first. HANDOFF_662 read the coincidence of
"starts at 5" and "the guard starts refusing at 5" as evidence that the
drain was gated behind the refused evaluation, and pointed the successor at
the guard. It was a detection threshold wearing the costume of an onset.
The guard is causal for the **deadlock only**, not for the leak.

Successor 19's asymmetric-alias lead (`install_resident_set` TP->PP) is
**not** the cause — correctly flagged there as unproven. The aliasing is
real and harmless, because the harvest dedupes by `id(batch)`.

### The fix, and why it is the correct rule rather than merely a working one

Unconditional assignment, which is what the non-PP loops have always done
(`self.last_batch = batch`, `batch` possibly None). Both loop families now
answer "what did the previous iteration run" the same way, and **"nothing"
is a real answer instead of a hole that preserves the old one**.

Containment is separate and was also owed: the guard's refusal was itself
the deadlock, because under strict purity only a flip to TP drains the
resident set. The catch site now repairs duplicated Reqs with the
scheduler's own `filter_batch` and re-asks, and logs the whole slot row
instead of a bare count — two boots were spent attributing that count.

### Evidence, desk

`test_pp_slot_last_batch_631.py` drives the **real** `_event_loop_pp_body`
taken unbound off the mixin; a model of the statement order would have been
circular, because the order IS the defect. `_LeakyRank` overrides one
method back to the pre-fix rule and nothing else:

| 25 cycles, 1 distinct request | resident entries | distinct | `last_mbs[0]` |
|---|---|---|---|
| PRE-FIX rule | **26** | 1 | STALE |
| shipped rule | 2 | 1 | None |

26 = 1 + 25 cycles, i.e. the metal law (+1 per round) reproduced exactly.
Full #631 family: **649 passed, 0 failed** (was 643; +6 new).

### Evidence, metal — and the instrument that matters

Boot 04:23Z, `RANK_MIB=31800,17400,17450`, `MAX_TOTAL_TOKENS=260000`,
CTX 393216, purity strict, POLICY auto. Load: 8 concurrent soak streams
against `max_running_requests=4` (`#queue-req: 5` — the trigger condition)
plus real qwen agent traffic through router 30099.

**The direct instrument is the carry count, not the absence of an alarm.**
`PHASE-FLIP-CARRY carried N resident request(s)` reports the actual size of
the resident set at every cutover. Under the old code it walked without
bound (5 -> ... -> 868447). In this window the only values that ever
appear are:

| carried | occurrences |
|---|---|
| 1 | 6 |
| 3 | 66 |

Bounded by `max_running_requests=4` across every flip. "No corruption
report" would only have proved the count stayed under the alarm threshold;
this proves the set is *stable*.

### The heavy phase, and why it was ended deliberately

The first 12 minutes ran **8** concurrent streams against
`max_running_requests=4` — over the design point on purpose, because
queueing pressure is what the leak needs. Result at 04:38:42Z:

| axis | value |
|---|---|
| flips | 228 |
| carried resident, all values seen | 1, 2, 3, 4 (ceiling is 4) |
| resident-set corruption reports | 0 |
| repairs performed | 0 |
| requests | 100, all HTTP 200 |
| agent requests (`POST /v1/messages`) | 44 |
| tracebacks | 0 |
| prefill batches with a CUDA graph | 0 of 335 (all PP, eager) |
| decode batches with graphs + `accept len` | 240 of 240 (all TP) |

**The leak verdict is complete at this point and does not depend on
anything that follows.** The pre-fix build wedged in 8 minutes under this
same load.

**The corridor, however, never plateaued**, and that is worth a successor's
attention because 662 reported a plateau at this pool:

| bucket | gpu0 | gpu1 (5090) | gpu2 |
|---|---|---|---|
| t-14..12 min | 4291 | 5610 | 3813 |
| t-10..08 min | 3927 | 5022 | 3451 |
| t-06..04 min | 3147 | 3902 | 2731 |
| t-02..00 min | 1945 | 2262 | 1629 |

Monotonic, roughly -260 MiB/min on gpu2 at the end, with no flattening over
5240 samples. Projected to cross the 1024 floor within ~2 minutes, so the
phase was ended before it did.

**Two things this establishes.** First, the allocator's high-water mark is
**sticky**: after the load stopped, idle free recovered only to
2745/2262/1789 against 5777/6372/4577 at boot. A window cannot be "given
back" its corridor by going quiet. Second, and this is the correction to
662: **at 260000 the corridor plateaus only for the workload it has been
shown.** 662 saw it flatten under soak-plus-agents; it does not flatten at
2x the design point. A plateau is a property of the load, not of the pool.

The corridor number therefore belongs to a window at the ACCEPTANCE load
(bs=4), held fixed, which is what the next section measures. Mixing the two
would have repeated successor 19's error 0.

### The acceptance-load window (bs=4), and why its allocator history is a FEATURE

Started 04:40:22Z on the same boot, 4 concurrent streams (the design point)
plus real agent traffic. The corridor flattens and stays flat:

| bucket | gpu0 | gpu1 (5090) | gpu2 |
|---|---|---|---|
| t-12..10 min | 1945 | 2262 | 1629 |
| t-08..06 min | 1945 | 2262 | 1627 |
| t-06..04 min | 1845 | 2162 | 1507 |
| t-04..02 min | 1825 | 2022 | **1467** |
| t-02..00 min | 1825 | 2022 | **1467** |

Two consecutive identical buckets, 443 MiB above the 1024 floor on the
binding card. Contrast the heavy phase, which never produced two equal
buckets in six.

**This window inherits the heavy phase's high-water mark, and that makes it
a WORST CASE rather than a contaminated one.** The mark is sticky, so free
memory here is strictly lower than a fresh boot at the same pool and load
would show. A pass under a pessimistic allocator history is therefore
stronger evidence than a pass from a clean boot, not weaker — which is the
opposite of the instinct, and is why the window was NOT restarted. The
restart would also have killed in-flight agent traffic for no gain
(successor 19's error 1).

The claim this window supports is consequently narrow and safe:
**260000 holds the corridor at the acceptance load even after a 2x-design
-point episode has permanently raised the allocator's mark.** It does NOT
establish the edge — 1467 MiB of headroom on the binding card says the edge
is higher, and finding it is the cheapest capacity work left.

### ...and then it breached. The plateau was not a plateau either

At 04:58:55Z, after ~18 minutes of the bs=4 window, the corridor collapsed:

| bucket | gpu0 (rank1) | gpu1 (rank0, 5090) | gpu2 (rank2) |
|---|---|---|---|
| t-10..08 min | 1825 | 2022 | 1467 |
| t-06..04 min | 1805 | 2022 | 1467 |
| t-04..02 min | 1725 | 1982 | 1387 |
| t-02..00 min | **1325** | **1382** | **1007** |

**gpu2 reached 1007 MiB — below the 1024 floor. This is a BREACH and the
window failed its corridor axis.** Window minima over 8214 samples:
1325 / 1382 / 1007.

The collapse tracks the agent request count (120 -> 140 over the same
minutes), which is 662's finding reproduced: the allocator's high-water mark
follows the largest request SHAPE it has seen, and agent turns keep getting
bigger. Three buckets of dead-flat readings preceded it. **This is the third
time in this chain that a flat corridor was read as a settled one.** Two
equal buckets are not a steady state; they are two equal buckets.

### The correction I owe on my own argument

I argued above that inheriting the heavy phase's sticky high-water mark made
this window a WORST CASE and therefore strong evidence. **That argument is
valid in one direction only, and I stated it as though it ran both ways.**

* a PASS under a pessimistic allocator history *would* have been strong;
* a **FAIL under one is weak**, because the pre-loaded mark is a live
  candidate for the entire cause.

So this breach does **NOT** falsify 260000 for a fresh boot. 662 measured
2167/2612/1809 at this pool on a clean allocator with heavy agents and saw
0 breaches in 3150 samples. What is established is narrower and still
useful:

**260000 does not survive a 2x-design-point episode followed by an hour of
growing agent context. A pool number is a property of a boot's whole load
HISTORY, not of the pool.**

That is a sharper statement of the sticky-mark finding, and it means every
capacity number in this document should carry the load history that produced
it, not just the load in flight at the time of reading.

## SUCCESSOR 20 / 2 — the corridor is bound by the 3080s OVERSHOOTING their budget

Measured on the T=190000 green run at 05:35Z, 30 minutes in:

| card | total | RANK_MIB | used | NVML free | total-used | overshoot |
|---|---|---|---|---|---|---|
| gpu0 3080 (rank1) | 20480 | 17400 | 18267 | 1789 | 2213 | **+867** |
| gpu1 5090 (rank0) | 32607 | 31800 | 29279 | 2810 | 3328 | -2521 |
| gpu2 3080 (rank2) | 20480 | 17450 | 18739 | **1317** | 1741 | **+1289** |

### Three findings, and the third relocates a constraint

**1. The carve-out is confirmed to the MiB.** `total-used` minus NVML `free`
is **exactly 424** on both 3080s. The memory law's warning is not a rule of
thumb: anyone sizing against `total-used` here reads 1741 MiB of headroom
where 1317 exists, an error of one whole safety margin.

**2. `RANK_MIB` is advisory in BOTH directions.** 662 established that a
too-small budget does not stop a too-large pool from overshooting. This adds
the other half: the 3080s exceed their assigned budget by 867 and 1289 MiB,
while the 5090 sits 2521 MiB UNDER its own.

**3. 662's "structural oddity" is real but is NOT the binding constraint.**
662 flagged that `RANK_MIB=31800` on a 32607 MiB card leaves 807 MiB — below
the 1024 floor by construction — and warned that every passing config held
only because the engine did not consume its whole budget. That is true, and
it does not bind: rank0 never approaches its budget, so the 5090 sits at
2810 MiB free, the most comfortable card in the rig. **The corridor is bound
by the two 3080s, which overshoot.**

The sharpest form: **rank2 is the tightest card while holding the SMALLEST
token share (20 of 74).** So the pressure on it is not KV — it is weights
plus per-rank overhead, which the token vector does not control. Tuning
`SGLANG_UNEVEN_TOKEN_VECTOR` or the pool therefore attacks the wrong term
for the card that actually binds.

### The lever that has never been tried

662 lowered `RANK_MIB` to 30500/16550/16500 **while the pool was still
460000** and the 5090 got WORSE (852 MiB free at idle), correctly concluding
"change the pool, not the budget". But that experiment holds only for the
combination it ran. **The untried cell is a LOW budget with a LOW pool**, and
specifically a lower budget on the two 3080s, which are the cards that
overshoot and the cards that bind. Nobody has run (low RANK_MIB x low pool).

### The corridor decays in STEPS, not as a drift

| bucket | gpu0 | gpu1 | gpu2 |
|---|---|---|---|
| t-35..30 min | 3093 | 3990 | 2603 |
| t-30..25 min | 2431 | 2988 | 1901 |
| t-25..20 min | 2089 | 2488 | 1579 |
| t-20..15 min | 2049 | 2488 | 1579 |
| t-15..10 min | 2049 | 2488 | 1577 |
| t-10..05 min | 2049 | 2486 | 1577 |
| t-05..00 min | 1809 | 2146 | **1317** |

**Four consecutive flat buckets and then a 260 MiB step.** This is the
mechanism behind the repeated "it plateaued and then it breached" reports:
the mark is a high-water function of request SHAPE, so it is flat exactly
until a bigger shape arrives. **A plateau of any length is not evidence of a
settled corridor — it is evidence that no larger request has arrived yet.**
Twenty minutes of flatness meant nothing.

## SUCCESSOR 20 / 3 — the T=190000 green run, in two segments

Boot 05:02Z, `MAX_TOTAL_TOKENS=190000`, `RANK_MIB=31800,17400,17450`
unchanged, CTX 393216, purity strict, POLICY auto, HEAD 8764b96589+.

### Segment A, 05:05:02Z -> 05:55:32Z (50.5 min): 4 soak streams + 4 agents

| axis | value |
|---|---|
| flips | 648 |
| carried resident, values seen | 1, 2, 3, 4 (ceiling 4) |
| corruption reports / repairs | **0 / 0** |
| tracebacks | 0 |
| health | 200 throughout |
| agent requests (`POST /v1/messages`) | 179 |
| soak requests | 79, all HTTP 200 |
| corridor time-series MIN | **1369 / 1646 / 1037** over 22167 samples |

**No breach — but 13 MiB of margin on gpu2, and still stepping down.** The
leak axis is spotless over 648 flips; the corridor is the whole story.

**The load was over-driven and I should name that plainly.** The acceptance
point is bs=4. Four soak streams PLUS four live agents is roughly twice
that, and it is the same over-drive I had already identified and corrected
once. Real agent traffic is the user's criterion; the synthetic soak is my
addition on top of it. So segment A measures "190000 at ~2x design point",
not the acceptance load.

### Segment B, from 05:55:32Z: agents only, i.e. the ACCEPTANCE load

Soak withdrawn, the four qwen agents left running, so the load is now the
one the green criterion actually names. This segment answers the question
that matters: **does the corridor hold at 190000 under real agent traffic?**

### The pattern across three pools, which is the useful part

| pool | load | time to trouble | corridor min |
|---|---|---|---|
| 260000 | 8 streams (2x) | ~12 min, no plateau | ended at 1629, falling |
| 260000 | 4 streams + agents | **BREACH at ~18 min** | 1325/1382/**1007** |
| 190000 | 4 streams + agents | 50 min, no breach | 1369/1646/**1037** |

Lowering the pool bought time — 18 minutes to 50 — but did not change the
SHAPE: the mark keeps stepping down as agent context grows, and each pool
merely sets how many steps fit before the floor. **That is evidence the pool
is the wrong knob for this failure.** The steps come from request shape, and
what converts a shape into resident bytes is the per-rank budget, not the
pool ceiling — which is exactly where SUCCESSOR 20 / 2 found the two 3080s
overshooting `RANK_MIB` by 867 and 1289 MiB while the 5090 sits 2521 MiB
under. **The untried low-`RANK_MIB` x low-pool cell is the indicated next
experiment, and this table is the argument for it.**

---

# SUCCESSOR 21 / 1 — THE BYTE LEDGER, AND THE TERM EVERY EARLIER LEDGER OMITTED

This section is the standing falsifier the user's override rule demands: no
capacity verdict counts unless the full delta between two configurations is
attributed by name, VRAM **and** host RAM. It is written because three
successors in a row produced a closure from a component analysis, and a
component analysis can silently omit a term while a closing balance cannot.

**It closes on a term nobody had itemised, and the omitted term is larger than
every term that had been argued about.**

## 1a. The static column, read from the live boot (pool 190000, HEAD 803222a339)

Instrument: `mem_ledger.flight_recorder`, which the boot script already arms
via `SGLANG_VRAM_FLIGHT_DIR=/spinning/flight_605`. Earlier handoffs said this
was unset; it is set — read `/proc/<pid>/environ`, not the log.

| term | rank0 (5090) | rank1 (3080) | rank2 (3080) | source |
|---|---|---|---|---|
| card total (NVML) | 32607 | 20480 | 20480 MiB | flight mark |
| weights arena `max(pp,tp)` | 14936 | 7924 | 9115 MiB | `PHASE-FLIP-BOOT TP stack built` |
| — PP layout image | 14936 | 6690 | 9115 MiB | `snapshotted ... MiB image` |
| — TP layout image | 13163 | 7924 | 7924 MiB | same |
| draft (MTP) weights, resident BOTH phases | 2058 | 1925 | 1925 MiB | `Load weight end` |
| mamba/GDN state pool | 1516 | 758 | 758 MiB | `Mamba Cache is allocated` |
| KV VMM arena, VA reserved | 3482 | 3174 | 2662 MiB | `KvVmmArena[...] ready` |
| — PP KV backing | 2944 | 1472 | 1472 MiB | `released the PP KV backing` |
| — TP KV backing | 2270 | 2040 | 1583 MiB | `KV Cache is allocated` (TP pass) |
| TP decode graph pool | 286 | 290 | 290 MiB | `capture_begin`→`capture_end` |
| PP graph pool | **0** | 0 | 0 MiB | eager by construction (661 §7) |
| NVML free at `boot_complete` | **7843** | **5764** | **5034** MiB | flight mark |

**The two KV pools are never both backed.** Traced through a full
pp→tp→pp cycle: `phase_flip_runtime.py:1489-1491` releases the source before
restoring the destination, `phase_flip_boot.py:718-734` asserts it at the
worst moment (TP allocated *and* graphs captured), and the two pools own
disjoint physical page sets through their own `KvVmmArena`. HANDOFF_663 §12
listed "two KV pools" and "duplicate graph pools" as unitemised leads; both
are now closed at **0 MiB**, by code and by the boot assertion.

**So the flip setup's static VRAM cost over plain TP3 is only the arena tail**
— `arena - tp_bytes` = **1773 / 0 / 1191 MiB** — plus the KV misalignment
(per-rank `max(layer_share, token_share)` sums to 1.1216 instead of 1.0).
Nowhere near the 6.4 / 4.5 / 3.5 GiB the conservation identity demands.

## 1b. The omitted term: post-boot allocator growth, and it is the whole gap

| | rank0 | rank1 | rank2 |
|---|---|---|---|
| NVML free at boot_complete | 7843 | 5764 | 5034 MiB |
| corridor minimum under load (HANDOFF_663 §9) | 1646 | 1369 | 1037 MiB |
| **decay after boot sizing** | **6197** | **4395** | **3997 MiB** |
| gap the conservation identity demands (663 §12) | 6554 | 4608 | 3584 MiB |

**The rows agree to within half a GiB on every card.** The mass that "exists,
is resident, and has a name" is not cold inactive-layout weights — those are
already at zero VRAM, because the arena is `max(pp,tp)` and the inactive
layout lives as a pinned host image. It is memory the process takes **after**
every boot-time itemisation has finished, which is exactly why five successive
boot-time analyses could not see it.

## 1c. What the growth IS, measured — the coefficient nobody had

`scripts/s21_scratch_ladder.py` sends one request per rung at a known prompt
length and reads the NVML floor from the 100 ms corridor sampler. Run
short-to-long on purpose: long-to-short allocates the worst case first and
reports zero for every later rung, which is true for that ordering and
useless.

Live instance, pool 190000, one request at a time:

| prompt tokens | marginal drop (MiB, card0/1/2) | cumulative floor |
|---|---|---|
| 2686 | 0 / 0 / 0 | 5017 / 6652 / 4347 |
| 5370 | 0 / 0 / 0 | 5017 / 6652 / 4347 |
| 10737 | 0 / 0 / 0 | 5017 / 6652 / 4347 |
| 21476 | 200 / 300 / 180 | 4817 / 6352 / 4167 |
| 42948 | 500 / 640 / 460 | 4317 / 5712 / 3707 |
| 85894 | 902 / 1320 / 924 | **3415 / 4392 / 2783** |

**~19-26 MiB of sticky allocator reserve per 1000 prompt tokens per card,
above a free tier of roughly 15k tokens.** It is monotone: the allocator never
returns the blocks on its own.

**This is the mechanism behind successor 20's law.** 663 recorded that "the
corridor decays in STEPS, not as a drift — a plateau of ANY length only means
no larger request has arrived yet" and that the misread had cost three
successors. That is precisely a high-water indexed by the longest prefill seen
so far. The law was right; the cause was never named, so it read as something
to be endured rather than something to be fixed.

## 1d. The falsifier that separates residue from peak

Asking the allocator to return its cached segments, on the same live instance,
with no reboot and no config change:

```
free BEFORE flush   3911, 4392, 2911 MiB
free AFTER  flush   6605, 7846, 5405 MiB     (+2694 / +3454 / +2494)
```

Full recovery to the boot-complete level, regardless of the accumulated
history. So the decay is **returnable**, and nothing in the phase-flip path
ever asks: `torch.cuda.empty_cache()` appears four times in
`phase_flip_boot.py` (:482, :543, :595, :747) and **zero times** in
`phase_flip_runtime.py`. The runtime flip has never reclaimed.

Re-running the ladder with a release before each rung separates the two
quantities, which must not be conflated:

| prompt tokens | in-flight dip (concurrent PEAK) | residue left behind |
|---|---|---|
| 21476 | 1640 / 1042 / 1112 | 240 / 320 / 220 |
| 42948 | 2118 / 1562 / 1572 | 480 / 640 / 462 |
| 85894 | **3042 / 2624 / 2456** | 922 / 1322 / 882 |

* the **peak** is live memory while the prefill runs (~30 MiB per 1000 tokens
  per card). No allocator call can return it, and a release-at-cutover must
  not be credited with it.
* the **residue** is what survives the request and every shorter request after
  it. That is what the rung reclaims, and it is why an hour-long run's
  corridor bears no relation to its first minute's.

## 1e. Why the KV pool was never the binding term, and the A/B that proves it

663 recorded pool 190000 holding at 1037 MiB and pool 260000 breaching at
1007 MiB. A 70000-token pool step costs ~1.1 GiB on rank 0 by the pool
arithmetic; the observed difference is **30 MiB**. Two pool sizes 37 % apart
produced the same corridor minimum to within noise, because in both cases the
allocator expanded until roughly the same amount of free memory was left.

Confirmed in code: **`--rank-gpu-memory-mib` is ADVISORY ONLY.** It becomes
`mem_fraction_static` (`server_args.py:10492-10512`), which is consumed once
in `_profile_available_bytes` to size the KV pool, and nothing enforces it
afterwards. `torch.cuda.set_per_process_memory_fraction` is called nowhere in
the tree. The VRAM dial (`managers/vram_dial.py`) is not a ceiling either — it
resizes the VMM-backed KV tail against a floor measured once and never
re-checked, i.e. it moves the one term the measurement shows is *not* growing.

**Therefore "260000 breached the corridor" was never a statement about 260000.**

## 1f. The host-RAM column, which the VRAM-only ledgers never had

The previous run died at 73 minutes on `oom_kill 9`, not on the corridor.

| | value |
|---|---|
| cgroup `memory.current` | 105.2 GiB |
| anon | 14.8 GiB |
| file | 89.7 GiB |
| **shmem** | **74.1 GiB** |
| swap | **0** |
| `/dev/shm` in use | 1.3 GiB |

The shmem is not `/dev/shm`; it is the schedulers themselves —
`RssShmem` 34.2 / 17.5 / 25.8 GB on PP0/PP1/PP2 — i.e. the **pinned weight
images**, `layout_pp.total_bytes + layout_tp.total_bytes` per rank, 58.3 GiB
by the boot log's own `images pinned` line. With no swap configured, pinned
shmem is unreclaimable, so ~89 GiB of a 117 GiB box is permanently spoken for
and the OOM killer is one long agent session away.

**Lever, recorded for whoever takes it:** the two images are READ-ONLY masters
— a flip only ever reads them into the arena, never writes them — and the same
bytes already exist on disk in the checkpoint. File-backing them converts
58 GiB of unswappable shmem into reclaimable page cache, at the cost of a page
fault on a cold flip. Not taken in this pass; the VRAM axis was the ordered
work.

## 1g. Which phase actually binds — and it is not what 663 assumed

`scripts/s21_phase_corridor.py` cuts the corridor series at the log's own
`event loop re-dispatch` instants and reports the minimum SEPARATELY per
phase, with a settle margin so the cutover's own transient is neither phase.
An aggregate minimum cannot answer this, and the answer decides whether any
spill is worth anything: an asset cold in the **binding** phase is worth its
full size, and one cold only in the other phase is worth exactly 0 MiB.

bs=4, pool 190000, 1961 samples over 4.5 min:

| card | pp min | tp min | seam min | binds |
|---|---|---|---|---|
| card0 (3080, rank1) | 5913 | 5017 | 5017 | **TP** |
| card1 (5090, rank0) | 7034 | 7378 | **6674** | **SEAM** |
| card2 (3080, rank2) | 5137 | 4349 | 4349 | **TP** |

663 §8 credited the arena tail at 0 MiB on every rank, on the reasoning that
the tail is idle only in each rank's non-binding phase. The tail is idle in
**TP**, and TP binds on both 3080s — and rank0's 1773 MiB tail is idle in TP
as well, where the seam (a TP-side event) sets the minimum. **That zero was
derived from an assumed binding phase, not a measured one. Re-derive before
reusing it.**

Second consequence, and it is new: **on the 5090 the deepest point of the
whole cycle is the cutover itself.** Any work on rank 0's corridor is work on
the seam, not on either phase's steady state.

## 1h. What this ledger licenses, and what it does not

It does **not** say >600k is reachable, and it does not say it is not. It says
the accounting that produced both previous verdicts was measuring the wrong
term, and it replaces that term with a measured one:

```
corridor_min(card) = boot_free(card)
                   - concurrent_peak_of_the_binding_regime(longest prefill)
                   - accumulated_residue(every longer prefill ever served)
```

The third term is reclaimable and is now reclaimed (SUCCESSOR 21 / 2). The
second is not reclaimable and scales at ~30 MiB per 1000 prompt tokens. The
first is what the pool competes for. Any future capacity claim must state
which of the three it is moving.

# SUCCESSOR 22 / 1 — THE MOVER WAS THE TRANSIENT, AND FIXING IT REMOVED THE BREACH

HANDOFF_664 section 13 traced the length-scaling VRAM transient to the
phase-flip KV mover and named the fix without building it. Built and
measured here. The one-line result: **at pool 400000 with the exact
111405-token request that breached the corridor for successor 21, the
corridor now holds with 1741 MiB to spare on the binding card, and the
request completes 3x faster.**

## 1a. Hermetic: what the mover held, before and after

Three-rank threaded flip, production row geometry (`row_nbytes = 2048 B`),
peak measured by a `TorchDispatchMode` probe over live tensor storage.
The probe EXCLUDES the persistent KV pools — an in-place op returns the
tensor it mutated, so `write_rows`' `k[idx] = ...` hands the probe the
destination pool itself (32 buffers, 20.6 MiB); counting them was a fifth
of the first reading and pure noise. `scripts/s22_mover_live_set.py`.

| direction | peak BEFORE | peak AFTER | plan floor | ratio before | ratio after |
|---|---|---|---|---|---|
| pp_to_tp | 39.4 MiB | **19.2 MiB** | 19.2 MiB | 2.05x | **1.00x** |
| tp_to_pp | 36.6 MiB | **20.2 MiB** | 19.2 MiB | 1.91x | **1.05x** |

The floor is `incoming + max(outgoing, local)`, computed from the plan and
never hardcoded. The mover now holds exactly what it owes.

Three changes, and the third is the one nobody had itemised:

1. `_pack_outgoing` fills one exact-size buffer per peer in place, instead
   of one tensor per layer plus `torch.cat` plus a checksum-appended copy
   (three copies of a peer's payload live at the peak);
2. `read_rows_into` is the pool-view primitive that makes that possible;
   `read_rows` is now built on it so the two cannot produce different
   bytes for the same rows;
3. **the outgoing buffers are released the moment `_exchange` returns.**
   They used to stay referenced through the local read, the backing swap
   and every write — the widest part of the move — making the peak
   `outgoing + incoming + local` when it only ever had to be
   `incoming + max(outgoing, local)`.

## 1b. Metal: pool 400000, the same trigger, only the code moved

Rebooted through `seam_scaling_reboot.py` (see 1d), same geometry
`PP_STAGE_RATIO=14,10,8` / `SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8` /
`MAMBA_SLOTS=12` / `RANK_MIB=31800,17400,17450` / CTX 393216 / purity
strict / policy auto / spill depth cache. HEAD f6e7f9803e.

| axis | s21, old mover (HANDOFF_664 §12) | s22, streamed mover |
|---|---|---|
| pool | 400000 | 400000 |
| prompt tokens | 111405 | 111405 |
| `#cached-token` | — | **0** (genuine full prefill, not a cache hit) |
| corridor min, rank1 (3080, binding) | **719 MiB — 305 UNDER the floor** | **2765 MiB — 1741 ABOVE** |
| corridor min, rank0 (5090) | 3106 | 5852 |
| corridor min, rank2 (3080) | 1151 | 2915 |
| request wall time | 68.5 s | **22.6 s** |
| `FLIP ABANDONED` | 0 | 0 |
| tracebacks / exits | 0 / 0 | 0 / 0 |
| spill rung fires | 42 / 42 flips | 24 / 24 flips |

Transient measured at 73008 prompt tokens: **1120 / 1346 / 982 MiB**, i.e.
**15.3 / 18.4 / 13.4 MiB per 1000 prompt tokens** against successor 21's
24 / 32 / 20. Going on from 73008 to 111405 prompt tokens added **zero**
further drop, which is the qualitative change: the term that tracked total
sequence length was the mover's, and it is bounded now.

The 3x wall-time gain is not a separate win. The mover allocated and freed
hundreds of MiB per flip and fired dozens of times inside one request;
removing two thirds of that allocator traffic is the same fix.

### READ THIS BEFORE COMPARING ANY PER-CARD TRIPLE IN THIS CORPUS

Every per-card triple in successors 20-21 is in **nvidia-smi INDEX order**,
which is `(rank1, rank0, rank2)` — NOT rank order. Confirmed two ways:
`SGLANG_RANK_CARD_UUIDS` on the live schedulers resolves rank0 to the 5090
(nvidia-smi index 1) and ranks 1/2 to the 3080s (indices 0 and 2), and
HANDOFF_664 §12a labels its own slopes "24 (rank1) / 32 (rank0) / 20
(rank2)" in that order. A successor who reads `719 / 3106 / 1151` as
rank0/1/2 will conclude the 5090 was the binding card. It was not.

## 1c. The accounting hole, and why the ORDER of the two fixes was load-bearing

`_staging_bytes` was `2 x outgoing + incoming` **with the local retained
leg missing entirely**. Every affordability refusal in this feature,
including the one that livelocked pool 500000, was computed from it.

It is now `incoming + max(outgoing, local) + one_layer_window`, where the
last term is the bounded per-layer gather that streaming did not remove
(it made it fixed at 1/L of a leg rather than proportional to the prompt).
Measured against the hermetic peak: predicted 20.2 MiB vs measured
19.2 / 20.2 — never short, never grossly over.

HANDOFF_664 §13c warned that shipping the formula fix ALONE would make the
livelock strictly more reachable, because the gate can only refuse and a
refusal does not drain the resident set it refused on. That warning is
now discharged rather than merely respected: streaming landed first, and
the honest budget (20.2 MiB) is SMALLER than the dishonest one it replaced
(30.2 MiB, over-reserving by 57%). On metal the same effect is visible per
flip — a flip that reserves 457 MiB under the new formula would have
reserved 650 MiB under the old one while actually needing less than either.

`test_phase_flip_staging_reserve_631` pinned the superseded formula BY
NAME (`test_outgoing_is_counted_twice_and_incoming_once`), which is how a
missing term looked deliberate for five successors. Rewritten with a
fixture in which the retained leg dominates — the geometry the old
expression could not see.

## 1d. A trap in the one tool that was mandated for capacity steps

`scripts/seam_scaling_reboot.py` read `/tmp/boot_cmdline.txt` and
`/tmp/boot_env.txt`. Against the running server those files were 10 hours
stale and differed in load-bearing ways:

| | capture file | live server |
|---|---|---|
| `pp-stage-ratio` | 2,1,1 | 14,10,8 |
| `rank-gpu-memory-mib` | 22700,11920,11970 | 31800,17400,17450 |
| `SGLANG_UNEVEN_TOKEN_VECTOR` | 28,26,20 | 14,10,8 |
| `PHASE_FLIP_PURITY` | **off** | **strict** |

So the tool prescribed for single-variable discipline would have booted a
different geometry with strict purity DISABLED and reported it as a
one-variable step — worse than a hand-built environment, because it looks
disciplined. It now reads the live process, stops it by PID as part of the
same invocation (capture and kill cannot be separated without losing the
baseline), takes arbitrary explicit substitutions, and writes a
baseline-vs-replay record per boot to
`/spinning/evidence-631/boot-captures/`.

## 1e. The `pp-stage-ratio` remap rule, which HANDOFF_664 §6 left unexplained

Successor 21 spent two boots optimising a split that the actuator did not
apply: `PP_STAGE_RATIO=15,10,7` came back as an achieved 16/9/7, `14,10,8`
mapped exactly, and the rule was left as an open question with the standing
instruction to verify the achieved split before spending a measurement.

The rule is quantisation to the full-attention block period. The model has
64 layers with one full-attention layer per 4, so a stage boundary can only
fall on a multiple of 4. `pp-stage-ratio` is in 32nds; the raw request is
`ratio / 32 * 64` layers, then rounded to a multiple of 4:

| stage ratio | raw layers | achieved `pp_layer_ratio` | full-attn per stage |
|---|---|---|---|
| 14,10,8 | 28, 20, 16 | **28, 20, 16** (exact) | 7, 5, 4 |
| 15,9,8 | 30, 18, 16 | **32, 16, 16** | 8, 4, 4 |

`14,10,8` maps exactly because 28/20/16 are already multiples of 4. Nothing
about `15,10,7` or `15,9,8` is; they land on the nearest legal boundary and
the sum is held at 64.

**Consequence for anyone planning a geometry step: the reachable splits are
not the 32nds, they are the multiples of 4 layers.** Asking for one layer
of relief on a stage gets you four or none. Read
`pp_layer_ratio=[...], pp_stage_ratio=[...]` out of the boot log — it prints
both, adjacent — rather than inferring the split from an arena size.

## 1f. The capacity curve, re-derived on the fixed mover — and it is now SEPARABLE

All triples below are in nvidia-smi index order = (rank1, rank0, rank2).

The single most useful consequence of the mover fix is that the two terms
finally separate. The transient is a property of the REQUEST and the
geometry; the idle floor is a property of the POOL. Measured at two pools
200000 apart, with the same 111405-token trigger:

| | rank1 | rank0 | rank2 |
|---|---|---|---|
| transient at pool 400000 | 1120 | 1346 | 982 |
| transient at pool 600000 | 1120 | 1366 | 926 |

Identical to within noise. With the old mover the two were entangled — the
staging peak scaled with the resident live set, which scaled with the pool
— which is why HANDOFF_664 §12a's model needed a separate fit per pool and
why §12 had to call the corridor and livelock bounds "two distinct bounds".
They are one bound and it is now additive:

```
  holds  iff  idle_free(pool) - transient(longest prefill) >= 1024 MiB
```

| card | idle-free slope, MiB per 1000 pool tokens | max pool holding the corridor |
|---|---|---|
| rank1 (3080) | 10.30 | **567,000**  <- binding |
| rank0 (5090) | 14.58 | 729,800 |
| rank2 (3080) | 8.36 | 617,100 |

## 1g. The geometry is quantised, and that is what blocks the 600000 target

`pp-stage-ratio 15,9,8` was the levelling step: give rank0's surplus a job
and relieve the binding card. Per 1e it lands on layers 32,16,16 — a
FOUR-layer move, not the one that was wanted — and the result is a clean
falsification of "just level it":

| pool 600000 | rank1 | rank0 | rank2 |
|---|---|---|---|
| idle free, 14,10,8 | 1809 | 4282 | 2149 |
| idle free, 15,9,8 | 3367 | 1490 | 2149 |
| floor under load, 14,10,8 | **689** (335 under) | 2916 | 1223 |
| floor under load, 15,9,8 | 2131 | **64** (960 under) | 1167 |

rank1 gains 1558 MiB and rank0 loses 2792 — and rank0's transient grows
too, because it went from 7 full-attention layers to 8. The card that had
1892 MiB of margin ends at **64 MiB free**, which is a near-OOM, not a
tuning miss.

**So the user's §10 surplus item is not a tuning nicety that nobody got
round to: at this layer count the surplus is UNREACHABLE with the layer
knob.** The reachable splits are the multiples of 4 layers, the two
candidates either side of balance are 7,5,4 and 8,4,4, and both leave one
card far from 1024 while another is at or below it. The token vector cannot
substitute, because after alignment each rank's KV is `max(KV_pp, KV_tp)`
with both equal — lowering a rank's TP share leaves it PP-bound and buys it
nothing while costing whoever receives the share.

Two named routes remain to >= 600000, and both are real work rather than
another boot:

1. **Spill rung 2 (draft weights).** 1925 MiB on rank1, resident in both
   phases, and the PP phase has no drafter at all. At this aligned geometry
   PP binds, so it pays its full size on the binding card — against a
   shortfall of 335 MiB. Needs the VA-stable carrier (`KvVmmArena`) because
   the dead module's `restore()` moves the draft addresses that the TP
   decode graphs bake. This is now by far the highest-value unbuilt item.
2. **Rebuilding the PP prefill KV layout**, which user spec item 2
   explicitly permits. It is the only lever that can break the
   `max(KV_pp, KV_tp)` symmetry that makes the token vector inert.

## 1h. EVERY TRANSIENT IN THIS CORPUS IS A LOWER BOUND, INCLUDING MINE

The mover's live set is `radix values UNION every resident request's
req_to_token[:seqlen]`. The union is over ALL resident requests. Every
transient measurement in this chain — successor 21's 2684/3584/2264, my
1120/1346/982 in 1b — was taken with the scratch ladder, which issues **one
request at a time**. So all of them measure a live set of one request, and
the deployment runs at bs=4.

Measured directly at pool 550000, geometry 14,10,8, idle free
2329/5014/2573:

| load | rank1 transient | rank1 corridor min |
|---|---|---|
| one 111405-token request (ladder, alone) | 1120 | — |
| bs=4 soak, mixed prompts, no long prefill yet | 1370 | 959 — BREACH |
| bs=4 soak + one 111405-token prefill on top | **1790** | **539** |

Per-card at the deepest point (idle free was 2329 / 5014 / 2573):

| | rank1 | rank0 | rank2 |
|---|---|---|---|
| corridor floor | **539** | 3262 | 1101 |
| transient | **1790** | 1752 | 1472 |
| single-request transient, same trigger | 1120 | 1346 | 982 |

**Concurrency adds 60% on the binding card** (1120 -> 1790), and the figure
was still deepening when the load was stopped, so treat 1790 as a floor on
the floor.

The breach arrived at +8 minutes, before the first scheduled 111405-token
prefill had fired. The pool had been sized from the single-request number
and the single-request number is 250 MiB short on this card.

**This is the sizing rule that was missing, and it is not a refinement of
§1f, it replaces its input:**

```
  holds  iff  idle_free(pool) - transient(TOTAL RESIDENT LIVE SET) >= 1024
```

where the total resident live set is bounded by
`max_running_requests x longest admitted prefill`, not by one request. A
capacity row quoting a pool without BOTH the concurrency and the longest
prefill it was measured at is not reproducible — which condemns the load
line on every capacity row in this document, mine included, that says only
"bs=4 soak" without the prefill length or only a prefill length without the
concurrency.

Note what this does NOT touch: the mover fix's result in 1b is a same-load
A/B (same ladder, same single request, same trigger, only the code moved),
so the 2.05x -> 1.00x and the 719 -> 2765 MiB corridor recovery stand
exactly as measured. What it invalidates is my extrapolation from that
measurement to a pool number for a bs=4 deployment.

### The corrected pool, and one thing the breach proves in the mover's favour

Re-solving the bound with the bs=4 transient (`slope 10.30 MiB per 1000
pool tokens on rank1`, `idle_free(550000) = 2329`):

| pool | rank1 idle | predicted floor | margin |
|---|---|---|---|
| 550000 | 2329 | 539 | **-485 (measured breach)** |
| 500000 | 2844 | 1054 | +30 |
| 480000 | 3050 | 1260 | +236 |
| **470000** | **3153** | **1363** | **+339** |

470000 is the green-run target: the transient was still deepening when the
load was stopped, so the margin has to absorb further high-water growth
rather than sit on the predicted value.

**And note what did NOT happen at 539 MiB free.** The instance stayed at
`/health` 200 with 0 `FLIP ABANDONED` and 0 tracebacks through a 485 MiB
corridor breach. Successor 21's pool 500000 livelocked at a rank that was
**13 MiB** short of its staging reserve. The staging bound and the corridor
bound have genuinely come apart: the corridor is now a budgeting question,
where before it was an availability one.

---

# successor 23 — two corrections to the sections above, and a measurement protocol

## 2a. CORRECTION to 1g ("the geometry is quantised"): there is no quantum

Section 1g closed the 600000 target on the claim that stage layer counts
quantise to multiples of 4, permitting only `[28,20,16]` and `[32,16,16]`.
That is this chain's FIFTH capacity closure and, like the four before it,
it is false.

`derive_pp_layer_split` (`distributed/utils.py:1481`, loop `:1540-1560`)
contains no literal 4 and no rounding to a multiple of anything. It rounds
the FULL-ATTENTION COUNT and clamps the proportional layer target into the
window between full-attention positions. "A whole number of full-attention
layers per stage" does not imply "a layer count divisible by 4".

Driving the real function on CPU over all 465 stage-ratio triples summing
to 32 (sanity-checked against the live server: `14,10,8 -> [28,20,16]`):

| quantity | value |
|---|---|
| distinct reachable `pp_layer_ratio` | **245** |
| triples refused (≥1 full-attn/stage guard, `utils.py:1573-1585`) | 52 |
| of the 245, NOT all-multiples-of-4 | **140** |
| effective granularity | **2 layers**, not 4 |

`15,10,7 -> [32,18,14]` and `16,9,7 -> [32,18,14]` — the ratio HANDOFF_664
§6 originally asked for. Section 1e's remap rule saw a quantum because
`target_full = round(cum/2)` with banker's rounding snaps only when
`cum ≡ 3 (mod 4)`, and `15,9,8` is that residue class. The snap there is
30 -> 32, TWO layers; the "four" is the gap between achieved geometries.

`--pp-layer-ratio` (`server_args.py:14641-14680`) bypasses the planner
entirely — any triple summing to 64 with ≥1 full-attention layer per stage
boots today. No consumer requires the alignment: KV cell sizing
(`pool_configurator.py:250-258`), the layer→row map (`memory_pool.py:3473`),
weight load (`utils/common.py:1986-2000`), dispatch (`qwen3_5.py:1412`),
`kv_reshard.py` and the graph runners all key off GLOBAL layer ids under a
half-open `start_layer <= i < end_layer`.

**Any capacity verdict resting on "only two geometries" must be recomputed.**

## 2b. CORRECTION to the capacity method: unflushed NVML-free readings are contaminated

HANDOFF_665 §13 concluded the pool lever is exhausted from a 26 MiB idle-free
move after a genuine ~750 MiB KV reduction. Measured here on the live IDLE
instance (0 running, 0 queued), `/flush_cache`
(`scheduler.py:6425`, `current_platform.empty_cache`):

| card | free before | free after | returned |
|---|---|---|---|
| rank1 (binding 3080) | 2079 | **3245** | +1166 |
| rank0 (5090) | 4876 | 6302 | +1426 |
| rank2 (3080) | 2327 | 3355 | +1028 |

Over a gibibyte per card of allocator cache is held at ZERO requests. The
freed KV bytes went there, invisible to the NVML free column.

**PROTOCOL: flush immediately before any idle-free reading, and state in the
row whether you did.** Every unflushed reading in this corpus is optimistic
by up to ~1.4 GiB, including the 37% budget-lever return in 665 §14.

`/flush_cache` also resets the radix cache, so it is a measurement
instrument, not a mid-benchmark action. It is refused while any request is
resident (`is_fully_idle()` gate), which is why it cannot be a runtime
corridor remedy.

## 2c. Why geometry alone does not close the corridor — the layer/token coupling

Corridor minima at pool 470000, nvidia-smi index order (rank1, rank0, rank2):
**381 / 3036 / 787**, total surplus above the 1024 floor **1132 MiB**.

From 665 §1.1, measured: a 4-layer move gained rank1 **+1558** (~390
MiB/layer) and cost rank0 **2792** (~698 MiB/layer). A layer is ~1.8x dearer
on rank0 because `SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8` gives it 14/32 of the
tokens. **So each layer moved onto the 5090 destroys ~308 MiB of total
corridor**, and rank0's 2012 MiB of slack absorbs 2 layers while rank1+rank2
need to shed 3. That, not a quantum, is why `15,9,8` overshot rank0 to 64 MiB.

The layer split and the token vector are INDEPENDENT knobs currently pinned
proportional to one another. Decoupling them — more layers AND fewer tokens
on the 5090 — is the lever that escapes the tax, needs no new code, and has
never been tried.

## 2d. Green-run row — 68 min, corridor HELD, 17 flips abandoned

Pool 380000, `RANK_MIB 31800,14000,15600`, geometry `[28,20,16]`, purity
strict, spill depth cache, NEXTN 3/1/4, `max_running_requests 4`.
Flushed idle baseline 4209/7618/4101 (idx order rank1, rank0, rank2).
Load history: 68 min of bs=4 soak + 4x111405-token prefill ladder +
decode probe + three qwen agent lanes through the router, concurrently.

| axis | value |
|---|---|
| corridor samples | 29740 over 68.1 min, 100 ms |
| minimum, idx order | **1215 / 3542 / 1349** |
| 1024 floor | HELD, worst margin **+191** |
| surplus above floor | 3034 MiB (too loose — the other half of the law) |
| flips | 834 LOG LINES = **278 flips** (139 each way) |
| FLIP ABANDONED | **51 lines = 17 events x 3 ranks** |
| tracebacks | 0 |
| prefill batches / with graph | 10989 / **0** (PURE) |
| purity refusals | 138 |
| decode batches | 1011, accept len **2.54** |
| decode `#running-req` | `{1:453, 2:270, 3:234, 4:54}` |
| largest prefill chunk | 2048, chunks > 2048: **0** (STATIC) |
| host `memory.peak` / `oom_kill` | 112.1 GiB / 9 (unchanged) |

Comparison against 665's configuration under comparable load makes the
trade explicit, and it is the finding of this row:

| | pool 470000, RANK_MIB ...,16200,16650 | pool 380000, RANK_MIB ...,14000,15600 |
|---|---|---|
| corridor min (binding) | **381 — breached by 643** | **1215 — held, +191** |
| FLIP ABANDONED | 0 | **17 events** |

**The corridor breach and the flip abandon are the same event.** Commit
`f4d8c1094e` chooses which one you get: it refuses a flip whose cache
credit is not collectable instead of letting the allocator take the
shortfall out of the user's reserve. The observed refusal —
`staging 3156 MiB needed but only 3074 spendable (driver free 4098,
allocator cache 246, reserve 1024)` — is short by 82 MiB, and the 246 MiB
of cache is what SURVIVED an `empty_cache`, so it could not have served a
3156 MiB contiguous buffer anyway.

**The binding term is now the 3156 MiB staging demand, not the pool and
not the geometry.** Both previous capacity levers are off the critical
path until staging bytes come down or the binding card gains ~500 MiB.

---

## 2e. THE SEAM IS WAVED — the 270k one-request livelock, removed (successor 24, 2026-08-10)

Commits `427db8f279` (the wave) and `510fb632a0` (its two constants).

### What was wrong

The flip swapped BOTH layouts' physical KV backing exactly once, at the
read/write seam. That forced every byte crossing the seam to be resident
at that one instant, so the staging demand was `sum(row_nbytes * n_rows)`
over the WHOLE plan — proportional to the resident live set and unbounded
in request length. The affordability gate was honest; it priced exactly
what the move allocated. **The defect was in the move.**

Under strict purity that does not degrade, it WEDGES: decode may only run
in TP, so a request that cannot be flipped never decodes, stays resident,
keeps the live set large, and the identical refusal repeats at ~1/s
forever. Health went 503 and only a reboot recovered.

### The fix

Split the seam into layer WAVES. Release the source layout's backing and
restore the destination's one wave at a time, so only one wave's payload
is ever staged. Each wave takes a proportional slice of every rank's own
layer block, which is what makes a wave's releases pay for its commits —
residency never rises above the resting layout while the staged bytes
fall by the wave count. The count is a property of the LAYER MAP (the
smallest stage's layer count), so what remains scales with pool geometry
rather than with the prompt.

`SGLANG_FLIP_SEAM_WAVES` overrides it; **1 reproduces the previous move
byte for byte**, which is what keeps this a one-variable A/B.

### Staging, before and after (rig fixture, map [28,20,16], 270k rows)

| | unwaved (1 wave) | waved (16) | at a FULL POOL |
|---|---|---|---|
| first cut (`427db8f279`) | 3856 MiB | 574 | 728 |
| constants priced (`510fb632a0`) | 3853 MiB | **418** | **550** |

The full-pool column is the closure: the live set cannot exceed the pool,
so **no request that fits this configuration can reach the refusal.**

The second commit went after two constants that dominated once the legs
were divided:

| term | first cut | priced | why |
|---|---|---|---|
| backing slack | 778 MiB | **242** | a flat "one layer of the bigger pool" over-reserved 3x; now computed by walking the wave plan and taking the worst boundary |
| gather window | 231 MiB | **~8** | `k[rows]` and the strided `.contiguous()` materialised the whole row list; now blocked (`SGLANG_FLIP_GATHER_ROWS`, default 16384) |

Over-reserving is not the safe direction. The gate's only action is to
refuse, and a refusal does not drain the resident set it refused on, so
invented headroom moves the livelock to a smaller request rather than
preventing one.

### Metal: the reproducer that wedged

`scripts/route_a_631_yarn_needle_probe.py --target-tokens 270000
--max-tokens 64`, bs=1, purity strict, pool 380000, map [7,5,4] over 16
full-attention layers → 4 waves.

| axis | successor 23 (unwaved) | successor 24 (waved) |
|---|---|---|
| FLIP ABANDONED | ~1/s, forever | **0** |
| flip committed | never | **yes**, at 270003 and 270012 live slots |
| health | 503 | **200 throughout** |
| staging reserved (per rank) | 3855 needed vs 3102 spendable | **2258 / 2192 / 1896 MiB** |
| flip time | — | 2020 / 2503 / 1784 ms |
| corridor min free | — | **2203 / 5154 / 2419**, 0 breaches of 1024 |
| recovery | reboot only | not needed |

Evidence: `/spinning/evidence-631/s24/{yarn_270k.json, yarn_270k.log,
corridor_270k.csv}`.

The metal figures are from `427db8f279`; `510fb632a0` cuts a further
~760 MiB off the per-rank staging, which the fixture table above prices
and the green run below measures.

### >262144 YaRN leg (spec item 4): GREEN

The same probe answers it, and it is the stronger reading. All three
planted needles were verified — including the DEEP one at ~95% depth,
i.e. past the 262144 native ceiling — at `prompt_tokens` 270031, and a
deep-only re-ask was answered from the same session at 270002 tokens
(268288 cached). The request **decoded**, which under strict purity can
only have happened in the TP layout. Prefill-only would not have produced
a token.

### A falsifier that failed, and what it taught

`TestSharedArenaReadsPrecedeWrites` models the two layouts ALIASING one
arena. It broke under the waved seam, correctly: waving interleaves reads
of one layout with writes of the other, so when the two overlay the same
bytes a wave's writes can land on rows a later wave has not read — the
#297 reads-before-writes hazard, reachable exactly here.

**An aliased arena and a waved seam are alternative capacity designs, not
a combination.** The runtime now detects overlap from the actual pointers
(`KvPoolView.overlaps`) and collapses the seam to a single wave with a
loud log. Anyone who later builds the aliased arena gets correct
behaviour and an explicit message saying staging will scale with the live
set again.

### Also

`flush_cache` now names WHICH clause of `is_fully_idle()` is false when
it refuses. The wedged instance refused the very flush its own abandon
message advertises, while every visible counter read zero, and the two
counters it printed could not say why. `Scheduler.idle_blockers()` is
diagnostic only — no caller branches on it.

## 2f. Green-run row — 64.5 min on the waved seam, ZERO abandons

Commit `510fb632a0`. Same recipe as 2d so the rows compare directly.

| axis | 2d (unwaved) | 2f (waved) |
|---|---|---|
| duration / samples | 68.1 min / 29740 | 64.5 min / 28114 |
| **FLIP ABANDONED** | **51 lines = 17 events** | **0** |
| flip DONE lines | 834 (278 flips) | 813 (271 flips) |
| tracebacks | 0 | 0 |
| corridor min, idx order | 1215 / 3542 / 1349 | **2699 / 5732 / 2831** |
| breaches of 1024 | 0 | 0 |
| worst margin | +191 | **+1675** |
| prefill batches / with graph | 10989 / 0 | 8049 / **0** (PURE) |
| purity refusals | 138 | 136 |
| decode batches / accept len | 1011 / 2.54 | 1647 / **2.734** |
| decode `#running-req` | {1:453,2:270,3:234,4:54} | {1:624,2:516,3:489,4:18} |
| host `memory.peak` / `oom_kill` | 112.1 GiB / 9 | 112.1 GiB / 9 (unchanged) |
| largest prefill chunk | 2048, none above | 2048, none above |

Agent traffic is in the log, not merely intended: 142 `/v1/messages` with
their 142 `/v1/messages/count` companions (the agent-SDK shape through the
router) and 124 `/v1/chat/completions`, on top of 119 `/v1/completions`
from the soak and ladder. Two qwen lanes ran real analysis work for the
whole window with no model override.

**The 17 abandons became 0 at a comparable flip count.** That is the
finding of this row. The corridor margin also went from +191 to +1675, but
only part of that is the smaller staging peak — this run had 8049 prefill
batches against 2d's 10989, so the load was lighter. Do not book the whole
1484 MiB as recovered headroom; measure it (section 2e, and HANDOFF_667
section 4 for the ledger).
