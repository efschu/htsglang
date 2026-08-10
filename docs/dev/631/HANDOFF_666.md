# #656 HANDOFF v26 — successor 23

Written 2026-08-10, tree `/spinning/wt-631-routea`, branch `feat/route-a-631`.

Read this before HANDOFF_665. Two of 665's load-bearing conclusions are
WRONG and are corrected here; both were wrong in the same way, and that
way is the most useful thing in this document.

---

## 1. MY ERRORS AND THE INHERITED ONES, ranked — read before any result

**0. I built the wrong fix first, and only the module's own docstring
stopped me.** I measured 1166 MiB of reclaimable allocator cache, found
that spill rung 1 reclaims only at the seam, and concluded the corridor
needed an INTRA-PREFILL reclaim keeper. I had written the design before
reading `phase_flip_spill.py`'s own note: *"It does not reduce the PEAK.
The transient a long prefill needs while it runs is LIVE memory, ~30 MiB
per 1000 prompt tokens per card, and no allocator call can give that back
while the prefill is in flight. What this returns is the RESIDUE."* A
previous successor had already separated peak from residue with a ladder
run. **The corpus contained the falsifier for my hypothesis and I nearly
spent a boot on it.** Read the docstring of the thing you are about to
duplicate.

**1. I enumerated the geometry space with a broken call and got
"UNREACHABLE" for every entry.** I called `derive_pp_layer_split` with a
`layer_types=` kwarg it does not take, inside a `try/except` that
swallowed the `TypeError`. Every target printed `UNREACHABLE`, which is
exactly the answer that would have confirmed the closure I was auditing.
I caught it only because I also printed one direct call outside the loop
and it raised. **A loop that swallows exceptions and a conclusion of
"nothing is reachable" are the same observation.** Always drive a known
input through the same path: `14,10,8 -> [28,20,16]` matching the live
server is what proved the second attempt sound.

**2. I nearly reported an instrument artifact as a defect.**
`PYTORCH_CUDA_ALLOC_CONF` is absent from `/proc/<scheduler>/environ` on
all three GPU-holding processes, which reads as "the allocator config
never reaches the cards". It is a `setproctitle` artifact — the
schedulers rename themselves to `sglang::scheduler_PP0`
(`scheduler.py:7967`), which clobbers the environ region. `PATH`, `HOME`,
`LANG` and `PYTHONPATH` are absent too. Checking for variables that
CANNOT be missing is the cheap test; do it before believing any
`/proc/*/environ` read on these processes.

**3. I dumped a full `server_args` line into my own context** by grepping
the boot log for `max_total_num_tokens` without bounding the output.
Every log read in this work must be `grep -n ... | cut -c1-200` or a
`sed` slice. The standing rule "never cat the server log" is not enough —
one grep line is 12 KB here.

---

## 2. HANDOFF_665's TWO WRONG CONCLUSIONS

### 2a. "The pool lever is exhausted" (665 §13) — UNSAFE, and the reason generalises

665 rebooted 550000 -> 470000, confirmed the KV pool really shrank
(per-rank `K size` came back in the 14:10:8 ratio, ~750 MiB less on
rank1), measured idle free moving **26 MiB** against a predicted 824, and
concluded the pool knob is dead and the per-rank budget is the only
corridor lever. All subsequent planning was redirected on that basis.

Measured here on the live IDLE instance (0 running, 0 queued),
`/flush_cache` — which calls `current_platform.empty_cache()` at
`scheduler.py:6425`:

| card | free before | free after | returned |
|---|---|---|---|
| rank1 (binding 3080) | 2079 | **3245** | +1166 |
| rank0 (5090) | 4876 | 6302 | +1426 |
| rank2 (3080) | 2327 | 3355 | +1028 |

Over a gibibyte per card of torch allocator cache is held **at zero
requests**. The 750 freed KV bytes did not vanish; they went into that
cache, where the NVML free column cannot see them. 665's 26 MiB measured
cache absorption, not the lever.

**Every NVML-free reading in this corpus taken without a preceding flush
is contaminated — including the 37% budget-lever return in 665 §14, which
is the number the whole remaining capacity plan was sized against.**

**PROTOCOL, from now on: `/flush_cache` immediately before any idle-free
read, and say in the row whether you did.** Note `/flush_cache` also
resets the radix cache, so it is a measurement instrument, not something
to do mid-benchmark.

### 2b. "600000 is structurally unreachable at both geometries the 4-layer quantum permits" (665 §6) — VOID

This was the chain's fifth capacity closure; four before it were false.
So is this one.

665 claimed stage layer counts quantise to multiples of 4 (the
full-attention block period), permitting only `[28,20,16]` and
`[32,16,16]`, and closed the user's capacity target on that basis.

**There is no multiple-of-4 quantum.** `derive_pp_layer_split`
(`distributed/utils.py:1481`, loop at `:1540-1560`) contains no literal 4
and no rounding to a multiple of anything. It rounds the FULL-ATTENTION
COUNT and clamps the proportional layer target into the window between
full-attention positions. "A whole number of full-attention layers per
stage" and "a layer count divisible by 4" are different constraints and
the second does not follow from the first.

Driving the real function on CPU over all 465 triples summing to 32
(reproduced by me — sanity `14,10,8 -> [28,20,16]` matches the live
server exactly):

* **245 distinct reachable `pp_layer_ratio` outputs**, 52 triples refused
  (all by the ≥1-full-attention-per-stage guard, `utils.py:1573-1585`).
* **140 of the 245 are NOT all-multiples-of-4.** Effective granularity is
  **2 layers**, not 4.
* `15,10,7 -> [32,18,14]` and `16,9,7 -> [32,18,14]` — the very ratio 664
  §6 asked for and 665 §1.1 recorded as an unexplained remap.

Why 665 saw a quantum: with `n_layers=64`, `n_full=16` and ratios in
32nds, `target_full = round(cum/2)`, and Python's round-half-to-EVEN
snaps only when `cum ≡ 3 (mod 4)`. They tested `15,9,8` — the one residue
class that snaps. `13,…` gives `[26,…]`, `17,…` gives `[34,…]`, neither a
multiple of 4. Their "four-layer move" is also a magnitude error: the
snap in their own example is 30 -> 32, **two** layers; the four is the gap
between two achieved geometries.

**And the planner can be bypassed entirely.** `--pp-layer-ratio`
(`server_args.py:14641-14680`) validates only `len == pp_size`, positive
integers, and `sum == depth`; `get_pp_indices` re-checks only length and
sum. So **any** triple summing to 64 with ≥1 full-attention layer per
stage boots today with no code change. `[29,19,16]`, `[26,18,20]`,
`[31,17,16]`, `[27,21,16]` were all driven through
`derive_pp_full_attn_layer_map`, `derive_pp_linear_layer_map` and
`validate_layer_map` successfully.

No consumer requires the alignment. KV cell sizing
(`pool_configurator.py:250-258`), the layer→row map
(`memory_pool.py:3473`), weight load (`utils/common.py:1986-2000`),
full-vs-linear dispatch (`qwen3_5.py:1412`), `kv_reshard.py`, and the
graph runners all key off GLOBAL layer ids filtered by a half-open
`start_layer <= i < end_layer`. Three alignment-sensitive sites exist —
MUSA (`musa/attention/flashattention_backend.py:66-77`), an NPU rope
warmup (`qwen3_5.py:1153`), and an SWA-gated off-by-one
(`model_runner.py:1570`) — none reachable on this path.

**Any capacity closure resting on "only two geometries" must be
recomputed.** The relaxation cost the spec asked me to price is ZERO;
the lever is already there.

---

## 3. WHY GEOMETRY STILL DOES NOT CLOSE THE CORRIDOR BY ITSELF

Having freed the geometry, I priced it, and it does not pay at this
token vector. Corridor minima at pool 470000 (nvidia-smi index order =
rank1, rank0, rank2): **381 / 3036 / 787**. Total surplus above the
1024 floor is **1132 MiB** — the verdict tool prints it.

From 665 §1.1, measured: a 4-layer move gained rank1 **+1558** (≈390
MiB/layer) and cost rank0 **2792** (≈698 MiB/layer). A layer is ~1.8x
more expensive on rank0 because `SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8`
gives it 14/32 of the tokens.

**So every layer moved onto the 5090 destroys ~308 MiB of total
corridor.** rank0's slack is 3036-1024 = 2012 MiB ≈ 2 absorbable layers,
while rank1 needs 2 shed (+780) and rank2 needs 1 (+390) — 3 layers to
place into room for 2. That is why `15,9,8` overshot rank0 to 64 MiB
free, and it is a property of the LAYER/TOKEN COUPLING, not of a quantum.

**The unexploited knob is therefore the TOKEN VECTOR, not the layer
split.** Audited, and the result is stronger than "it is possible":
**the two knobs are ALREADY fully decoupled, and the live configuration
is BACKWARDS relative to the design intent.**

* No code ties them. A grep for any cross-reference between
  `pp_layer_ratio`/`pp_stage_ratio` and
  `SGLANG_UNEVEN_TOKEN_VECTOR`/`cp_token_ratio` returns nothing. They are
  validated independently — the layer ratio against `num_hidden_layers`
  (`server_args.py:14679-14700`), the token vector as positive integers of
  length `dcp_size` (`distributed/utils.py:540-551`).
* Different axes: the layer ratio places WEIGHT LAYERS on PP stages; the
  token vector sets each rank's KV TOKEN-SLOT share via the owner rule.
* **`parse_flip_token_vector()` (`phase_flip_boot.py:84-113`) documents why
  they SHOULD differ: weight shards follow COMPUTE (the 5090 takes the
  biggest share), token slots follow REMAINING MEMORY (the 5090 takes the
  SMALLEST token share precisely because its weight shard is largest).**
  The code expects divergence and logs it (`phase_flip_boot.py:465-474`).
* The live config sets the token vector EQUAL to the stage ratio
  (`14,10,8`), giving the 5090 the largest share on BOTH axes. **That is
  the direct cause of the ~698 vs ~390 MiB/layer asymmetry above.**
* It drifted there by default: `SGLANG_UNEVEN_TOKEN_VECTOR` overrides the
  flip vector when set and DEFAULTS to it when unset
  (`phase_flip_boot.py:118-142`). It is GCD-reduced, so `14,10,8` is
  effectively `[7,5,4]`.
* Why the denominators differ (32 vs the flip vector's 64): the flip vector
  doubles as the weight-shard plan (`rank_tp_ratio`), so its sum must divide
  the sharded dimensions — `hidden_size 5120 % 64 == 0`
  (`distributed/utils.py:869-878`). The token vector carries no such
  constraint; only its reduced ratio matters.

**So the move is a single ENV change, no code: give the 5090 MORE layers and
FEWER tokens.** It attacks the 308 MiB/layer tax at its source, it is what
the module was designed for, and nobody in this chain has tried it.

---

## 4. WHAT I SHIPPED

`f4d8c1094e` — `_staging_affordable` materialises its cache credit
before spending it. The guard read
`usable = cached_free + max(0, driver_free - reserve)`, where
`cached_free` is the SUM of every free block; a 457 MiB staging buffer
cannot be cut out of 1166 MiB scattered across small ones. When the cache
cannot serve the request the allocator goes to the driver instead — either
spending the corridor or raising the `OutOfMemoryError` out of
`kv_reshard._exchange` that the file records as having killed all three
ranks. Now: when the verdict must lean on the cache, hand the cache back
first, then judge against `driver_free` alone.

Strictly no more permissive than what it replaced, so it cannot introduce
an OOM the old code refused. **This does NOT contradict 664 §11a** ("run
rung 1 before the affordability test is worth ZERO") — that is right
about the ABANDON verdict. The value is the other two: a flip whose cache
credit is not collectable is refused instead of blessed, and the corridor
is restored on the ABANDON path, which rung 1
(`phase_flip_runtime.py:1517`) never reaches because it sits downstream
of the refusal.

Red-first: 4 of 5 new tests fail on the reverted source (the fifth
asserts NO reclaim and passes either way). Family suite
`scripts/run_631_flip_family.sh`: **669 passed, 0 failed**.

---

## 5. ENFORCEMENT (`set_per_process_memory_fraction`) IS A TRAP — do not build it blind

664 §6 and 665 called a per-rank enforced memory fraction the top unbuilt
item, on the argument that it would make the budget lever 1:1. Audited:

* It is genuinely called **nowhere** in the tree (9 hits, all in docs).
* **It would be compared against a VIRTUAL extent.** The fork's own
  measurement, `phase_flip_spill.py:254-258`: under
  `expandable_segments:True`, torch's reserved figure "counts a VIRTUAL
  extent and was observed at 36910 MiB on a 32607 MiB card, so it cannot
  be compared to a physical budget at all." A fraction ceiling would
  refuse allocations long before the physical budget is touched.
* The denominator differs from the flag's. `--rank-gpu-memory-mib`
  divides by **NVML total** (`server_args.py:10554-10556`);
  `torch.cuda.set_per_process_memory_fraction` multiplies by the **CUDA**
  total, and `scripts/probe_652_device_total.py:5-11` records the 5090
  showing ~21.9 GiB reachable against nvidia-smi's 32607.
* The ceiling bounds torch-allocator bytes only — not the CUDA context,
  cuBLAS/flashinfer workspaces, or NCCL (the live env has
  `NCCL_CUMEM_ENABLE=0`, so NCCL allocates outside the torch allocator).
* **If you ever do enforce it, you must change `_staging_affordable` in
  the SAME commit.** Its `max(0, driver_free - reserve)` term is exactly
  the bytes that come from the driver as NEW segments — outside the
  ceiling. The guard would authorise spending them and the allocator
  would refuse, converting today's graceful abandon into the
  three-rank-killing OOM.
* `garbage_collection_threshold` is a complement, not a substitute: its
  denominator IS the fraction ceiling
  (`c10/core/AllocatorConfig.h:202-207`), so with the default 1.0 it only
  fires near 100% of the card.

Torch 2.11.0+cu130 does release cached blocks and retry before raising
(disassembly of `DeviceCachingAllocator::malloc` in `libc10_cuda.so`
shows the `release_cached_blocks` / `garbage_collect_cached_blocks`
ladder with two further `alloc_block` calls after it; `memory.py:272`
documents `num_alloc_retries`). Useful, but it does not rescue the
virtual-extent problem.

---

## 6. STATE LEFT BEHIND

See §7 for the green-run outcome. Serving geometry unchanged at
`pp_stage_ratio=[14,10,8]` -> `pp_layer_ratio=[28,20,16]`, purity strict,
policy auto, spill depth cache, CTX 393216, `max_running_requests 4`,
`MAX_MAMBA_CACHE 12`, NEXTN spec 3/1/4.

The 470000 + `RANK_MIB 31800,16200,16650` configuration 665 left armed is
**not shippable**: 30.2 min of 100-ms corridor samples give minima
381/3036/787, rank1 breaching by 643. 665 predicted a floor of ~957 and
called it "one flag change away"; it was out by an order. 246 flips
(123 each way), 0 `FLIP ABANDONED`, 0 tracebacks, purity PURE (2442
prefill batches, 0 with a CUDA graph, 39 purity refusals), accept len
2.26 over 378 decode batches, host `memory.peak` 112.1 GiB, `oom_kill` 9
(unchanged precedent).

Decode is confirmed **not** bs=1 — the histogram over that window is
`{1: 309, 2: 69}`, and 665's own probe saw `{1:15, 2:42, 3:90}`. The
inherited "bs=1 instead of 4" suspicion is dead; what remains open is
in-phase decode tok/s versus plain TP3, which
`scripts/s22_decode_probe.py` separates and which nobody has run inside a
load window.

---

## 7. THE GREEN RUN — 68 minutes, corridor HELD, but NOT green

Configuration (two variables moved from 665's, recorded as such):
`--max-total-tokens 380000`, `--rank-gpu-memory-mib 31800,14000,15600`,
geometry UNCHANGED at `pp_stage_ratio [14,10,8] -> pp_layer_ratio
[28,20,16]`, purity strict, policy auto, spill depth cache, CTX 393216,
NEXTN 3/1/4, `max_running_requests 4`. Pool held at 380000, not clamped.

Flushed idle baseline (new protocol), idx order (rank1, rank0, rank2):
**4209 / 7618 / 4101** — against 3245/6302/3355 at 665's configuration.

Load, all at once for 68 min: 100-ms corridor sampler, `s20_soak` at
bs=4, the 4x111405-token long-prefill ladder, `s22_decode_probe`, and
three qwen agent lanes through the router with NO model override.

| axis | result |
|---|---|
| corridor samples | 29740 over **68.1 min** |
| minimum, idx order | **1215 / 3542 / 1349** |
| floor 1024 MiB | **HELD**, worst 1215, margin +191 |
| surplus above floor | 3034 MiB — the OTHER half of the law, still too loose |
| flips | 834 LOG LINES = **278 flips** (139 each way) — see §7c |
| **FLIP ABANDONED** | **51 lines = 17 events x 3 ranks** |
| tracebacks | 0 |
| prefill batches | 10989, **0 with a CUDA graph** (PURE — PP only) |
| purity refusals | 138 (`prefill cannot run in tp`, the gate acting) |
| decode batches | 1011, all carrying accept len |
| accept len | **2.54** mean over 1011 |
| decode `#running-req` | `{1:453, 2:270, 3:234, 4:54}` — reaches bs=4 |
| host RAM `memory.peak` | 112.1 GiB, `oom_kill` **9** (precedent unchanged) |

**IT IS NOT GREEN, AND THE REASON IS MY OWN COMMIT.** The abandon count
was **0** at the 34-minute midpoint and 51 at the end. The refusal reads:

```
staging 3156 MiB needed but only 3074 MiB is spendable
(driver free 4098 MiB, allocator cache 246 MiB, reserve 1024 MiB kept free)
```

`4098 - 1024 = 3074` is exactly my new formula. The superseded formula
would have added the 246 MiB cache credit, got 3320, and blessed it.

**But the refusal is correct, and this is the important part.** That 246
MiB is what SURVIVED an actual `empty_cache` — my code reclaims, re-probes,
then logs — so it is genuinely stranded and could never have served a 3156
MiB contiguous buffer. The old code would have gone to the driver for the
82 MiB shortfall, straight into the user's reserve.

**So the corridor breach and the flip abandon are THE SAME EVENT, and this
commit chooses which one you get.** 665's run: corridor breached to 381,
0 abandons. This run: corridor held at 1215, 17 abandons. Under the user's
hard "never breach" law the trade is the right way round, but it is a real
availability cost and must not be reported as a clean pass. 665 §9 said
"the corridor and the staging bound have come apart"; this re-couples them
deliberately.

No livelock: the instance held `/health` 200 throughout, and flips
continued in both directions after the abandons.

**What is still needed for green.** The 3156 MiB staging demand against
~4100 MiB of driver-free is the binding term now — not the pool, not the
geometry. Either reduce staging bytes or buy the binding card ~500 MiB
more headroom, then re-run this exact recipe.

### 7b. Decode decomposition — INSTRUMENT FAILED, do not quote its duty cycle

`s22_decode_probe.py --concurrency 4 --max-new 600 --rounds 3`, run
INSIDE the window, completed: 12 requests ok / 0 failed, 7200 output
tokens, 329.5 s wall, **21.9 aggregate tok/s**, per-request median 5.7
(min 4.8, max 8.3).

It also printed **"0.0 s inside TP (0% TP duty cycle)"** and therefore
produced NO in-phase number. **That reading is false on its face** — 1011
decode batches ran under strict purity, which forbids decode outside TP.
The probe's TP-window detection is broken; fix it before trusting any
in-phase figure. The open question (in-phase decode tok/s versus plain
TP3) is therefore STILL unanswered after six successors.

The one decode question that IS settled: **decode is not bs=1.** The
histogram over 1011 batches is `{1:453, 2:270, 3:234, 4:54}`.

---

## 8. WHAT I WOULD DO NEXT, in order

1. **Decouple the token vector from the layer split** (§3). Highest value,
   no new code, and it is the only lever that escapes the 308 MiB/layer
   levelling tax. Give the 5090 MORE layers and FEWER tokens; re-price the
   corridor with a flush-before-read at each step.
2. **Re-attack the user's capacity target against the 245-geometry list**
   (§2b), now that the closure is void. Use `--pp-layer-ratio` for
   1-layer granularity where the ratio planner's 2-layer steps overshoot.
3. **Re-derive the budget-lever return rate with flushed readings**
   (§2a). The 37% figure is contaminated and every plan sized against it
   inherits the error.
4. **Decode decomposition** — `scripts/s22_decode_probe.py --concurrency 4
   --max-new 600 --rounds 3`, INSIDE a load window. Note decode logging is
   gated on `decode_log_interval` (default 40), so short generations
   produce zero decode evidence.
5. **Prefill chunk A/B including the dynamic arm.** Still unmeasured, and
   **665 §16's "the arm is free" is WRONG — see §10.**

---

## 10. CORRECTION to 665 §16: the dynamic-chunking arm is NOT free

665 §16 withdrew its own VRAM objection to `--enable-dynamic-chunking`
after verifying that `max_prefill_buffer_tokens()` has exactly one
consumer (`eager_runner.py:114`) where the 16384 is dominated ~29x by
`max_total_num_tokens`. That verification is correct and I reproduce it.

**It checked the wrong quantity.** The number that varies at runtime is
`chunked_prefill_size`, and that has TEN memory-sizing consumers, of
which only `eager_runner.py:114` goes through the inflating accessor:

| file:line | what it sizes |
|---|---|
| `flashinfer.py:122-123` | FlashInfer MoE workspace |
| `gdn_backend.py:76` | GDN attention chunk |
| `n_gram_embedding.py:85` | n-gram embedding buffer |
| `indexer_topk.py:34` | indexer topk buffer |
| `routed_experts.py:67` | routed-experts buffer |
| `expert_distribution.py:391` | expert distribution (`x8`) |
| `triton_symm_mem_ag.py:438` | Triton symmetric all-gather |
| `mhc.py:580` | MHC prewarm buckets |
| `kt_ep_wrapper.py:91` | KT EP config |

`max_prefill_buffer_tokens()` (`server_args.py:14213-14226`) inflates to
`ceil(chunked * 1.25)` under dynamic chunking; those nine read
`server_args.chunked_prefill_size` RAW. And the runtime dynamic size CAN
exceed the base — the quadratic solver at
`scheduler_pp_mixin.py:2684-2702` produces values above `base_chunk_size`
at low `history_len`, smoothed toward it with coefficient 0.75
(`environ.py:586`). The boot profiler is more concrete still: it runs
**128 real prefills** at up to `chunked_prefill_size * 1.25`
(`scheduler_pp_mixin.py:1578-1584`) — 2560 tokens against buffers sized
for 2048.

**The error shape is 665's own, one level down.** 665 §16's lesson was
"a quantity growing is not a cost; find the consumer first." They found
the consumer of the quantity they were TOLD grew, and never grepped the
quantity that actually varies. Before booting the arm, size those nine
buffers at 1.25x or pin `chunked_prefill_size` so the inflation cannot
exceed them.

Engagement evidence, verified (`scheduler.py:1531` `pp_size > 1` gate,
`:1542` self-disable):
* success: `[PP Dynamic Chunk] [PP{rank}] Predictor ready (quadratic)`
  (`scheduler_pp_mixin.py:1738`), `[ChunkSizePredictor] Fitted
  coefficients:` (`:2627`)
* failure: `[PP Dynamic Chunk] Failed to profile prefill latency:`
  (`scheduler.py:1538`)
* the per-chunk `Predicted chunk size:` line (`:1770`) is `logger.debug`
  and will not appear at the default level.

Confirmed: the FIRST chunk of every request always uses the static size
(`scheduler.py:5072`, dynamic applied only when `chunked_req is not
None`), so an A/B must use prompts long enough for later chunks to
dominate. And a static run shows many distinct `#new-token:` values (117
in my 68-min run) — the discriminator is a chunk EXCEEDING
`chunked_prefill_size`, which my run confirms at **0**.

---

## 9. STANDING OPERATIONAL NOTES

* Do NOT use TCP connection count to judge idleness — the router at 30099
  holds persistent keep-alives. Use `sglang:num_running_reqs` +
  `sglang:num_queue_reqs`.
* Boots ONLY through `scripts/seam_scaling_reboot.py` (reads the live
  process, stops it by PID, records a baseline-vs-replay diff to
  `/spinning/evidence-631/boot-captures/`). It warns when more than one
  variable moved; believe the warning.
* `PYTHONPATH=/spinning/wt-631-routea/python` for every test and tool, and
  the live server carries it too — verify with
  `tr '\0' '\n' < /proc/<launcher pid>/environ | grep PYTHONPATH` on the
  LAUNCHER, never on a scheduler (§1.2).

---

## 7c. THE DECODE PROBE'S BROKEN INSTRUMENT — root-caused and FIXED, plus a counting error of mine

`s22_decode_probe.py` reported **"0.0 s inside TP (0% TP duty cycle)"** on a
run whose log holds 1026 `PHASE-FLIP DONE` lines. Zero is impossible under
strict purity, where decode may ONLY run in TP — the instrument was
contradicting a boot-enforced invariant, which is the cheapest kind of
falsifier and should have been checked the moment it printed.

**Root cause.** `tp_seconds()` grepped
`[0-9-]+ [0-9:]+\] PHASE-FLIP DONE (pp_to_tp|tp_to_pp)`, requiring `]`
immediately after the timestamp. The real line carries a RANK TAG:

```
[2026-08-10 10:47:30 PP2] PHASE-FLIP DONE pp_to_tp
```

So the grep matched NOTHING, `events` stayed empty, `in_tp` stayed False,
and the function returned 0.0. Fixed to accept the rank tag.

**A second bug behind the first, and it is the one that caught me too.**
One flip is logged ONCE PER RANK. My first fix collapsed on
`(timestamp, direction)` — and that is WRONG: the log has one-second
resolution and the three ranks straddle second boundaries, so 1026 lines
became 560 "distinct" events against ~342 real flips. The fix filters to a
SINGLE rank, which is exact by construction. Verified: 1026 lines ->
**342 events, 171 pp_to_tp / 171 tp_to_pp**. The perfect balance is the
sanity check — flips must alternate.

**CORRECTION TO MY OWN §7 NUMBERS.** I reported "834 flips (417/417)" from
the verdict tool. That is 834 LOG LINES across three ranks =
**278 flips (139 each way)**. `s22_green_verdict.py:62` counts lines, not
events, and every flip count in this corpus quoted from it is 3x high. I
de-duplicated the ABANDON count correctly (51 lines -> 17 events) in the
same session and then failed to apply the identical correction one row
above it. The abandon RATE is unchanged (17/278 = 6%), but the absolute
flip figures were wrong.

`s22_green_verdict.py` still line-counts; a successor should apply the same
single-rank filter there.

**The decode decomposition is now unblocked** — the probe was the blocker,
not the workload. Re-run it inside a load window and the in-phase number
will exist for the first time in this chain.

---

## 11. THE LIVELOCK, REPRODUCED DETERMINISTICALLY BY ONE REQUEST — the headline

I ran the YaRN acceptance leg empirically because a code audit is not a
measurement. It wedged the instance, and the mechanism is now fully
documented with a **one-request reproducer**, which this chain has never had.

**Reproducer:** `scripts/route_a_631_yarn_needle_probe.py --target-tokens
270000 --max-tokens 64` against pool 380000 /
`RANK_MIB 31800,14000,15600` / geometry `[28,20,16]` / purity strict.
Single request, bs=1, server otherwise idle.

**What happens, in order:**

1. The 270032-token prompt **prefills successfully in PP** —
   `kv_committed_len=270031`. So YaRN above the 262144 base genuinely
   works, and §7's audit verdict is confirmed for the PREFILL half.
2. The `pp_to_tp` flip is then abandoned:

```
staging reclaim: driver free 4126 -> 4126 MiB (+0 returned), 226 MiB still
                 cached, reserve 1024 MiB, staging needs 3855 MiB
FLIP ABANDONED (pool too small for the live set): pp_to_tp.
   staging 3855 MiB needed but only 3102 MiB is spendable
```

3. **Under strict purity decode may ONLY run in TP.** An abandoned
   `pp_to_tp` means the request can never decode, so it stays resident, so
   the live set stays large, so staging still needs 3855 MiB. Forever.
4. `/health` goes 503; a trivial 8-token completion times out at 90 s.
5. Killing the client drains the request — `num_running_reqs 0`,
   `num_queue_reqs 0` — **and the livelock continues**, abandon lines
   growing ~1/second (408 -> 432 in 25 s). The RESIDENT PREFIX CACHE still
   holds the live set.
6. `/flush_cache`, the remedy the abandon message itself advertises,
   returns **HTTP 400**. `is_fully_idle()` is False while the metrics
   report zero running and zero queued — a state/metrics divergence.
7. No external remedy exists. Only a reboot recovered it (664 §11c, now
   confirmed on a second, cleanly reproducible case).

**MY COMMIT NEITHER CAUSED NOR PREVENTED THIS.** The reclaim returned
**+0 MiB**; the superseded formula would have offered `226 + 3102 = 3328`,
still short of 3855. It livelocks identically on the old code. This is a
pre-existing defect that a single long request exposes on demand.

### What it means for the program

* **STAGING BYTES SCALE WITH THE RESIDENT LIVE SET** — 3156 MiB under the
  68-min mixed load, **3855 MiB** for one 270k request. There is therefore a
  request length beyond which the flip can NEVER afford to carry the live
  set, and under strict purity that length is a **hard context ceiling that
  wedges the instance rather than degrading**.
* **This bounds user acceptance item 4 (bs=1 + YaRN above 262144).** The
  model config permits 393216, the rope cache really covers it, and the
  prefill completes — but decode is unreachable at 270k on this budget.
  Prefill-only success is NOT the acceptance item.
* **It is the top defect, ahead of every capacity item.** A wedge that any
  long request can trigger outranks corridor tuning.
* **Two candidate fixes, neither built.** (a) Make the abandon path
  RECOVERABLE from inside the scheduler — retract or preempt the resident
  request, or drop its prefix cache, then retry; 664 §11c argued this and
  the reproducer now makes it testable in minutes. (b) Bound staging bytes
  so they cannot scale without limit with the live set (chunk the move).
* **Fix `is_fully_idle()`/metrics divergence** so `/flush_cache` is
  available in the state that needs it — that alone would have converted
  this wedge into a recoverable event.

**A REGRESSION TEST EXISTS NOW.** Any future claim that the livelock family
is closed must run the 270k reproducer above. The three previously "known"
instances were all argued from logs; this one is a command.
