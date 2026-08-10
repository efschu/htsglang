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
split.** Layers and tokens are independent
(`SGLANG_UNEVEN_TOKEN_VECTOR` vs `--pp-stage-ratio`/`--pp-layer-ratio`),
and they are currently pinned proportional to each other (14,10,8 and
28,20,16). Decoupling them — more layers AND fewer tokens on the 5090 —
is the move that can level the corridor without the 308 MiB/layer tax.
**Nobody in this chain has tried it.** That is the highest-value
unexplored lever and it needs no new code.

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

## 7. THE GREEN RUN

Status recorded in §7a below at the end of my session. The recipe in 665
§11 is sound EXCEPT that its pool/budget numbers do not hold the
corridor; use the configuration named in §7a.

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
5. **Prefill chunk A/B including the dynamic arm.** Still unmeasured. The
   arm is free (665 §16, verified). Two traps: the discriminator is a
   chunk EXCEEDING `chunked_prefill_size` (a static run already shows ~43
   distinct sizes), and the FIRST chunk of every request always uses the
   static size (`scheduler.py:5072`).

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
