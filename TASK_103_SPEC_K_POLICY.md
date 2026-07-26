# Task #103 — speculation/split axis on the Qwen3.6-27B-FP8 working arm

Working arm under test: Qwen3.6-27B-FP8, TP=3 uneven DCP, flashinfer,
MTP NEXTN (steps/topk/draft-tokens = 3/1/4), CUDA graphs ON, token ownership
vector pinned `SGLANG_UNEVEN_TOKEN_VECTOR=6,5,5` (#188: pin the budget when
comparing).

Bar to clear: **> 5 % decode, single AND dual session**, without quality loss
and without max-KV falling more than 10 %.

Harness: `/spinning/r3val/noise_floor.py` (per-block driver, also used per
arm), `/spinning/r3val/arm_measure.sh` (per-arm protocol),
`/spinning/r3val/cmp_arms.py` (two-arm comparison with detection limit).
Raw per-block records: `/spinning/r3val/logs/nf_*.jsonl` (appended
incrementally, so a killed run keeps its points).

---

## 0. Measurement methodology — established BEFORE any arm was judged

This section exists because the first attempt at this campaign was about to
report a 1 % effect as real. It is not reportable on this rig.

### 0.1 Clock pinning is impossible here

`nvidia-smi -lgc` (and `-pm`) are **refused by the driver** on all three
GeForce cards, as root:

```
The current user does not have permission to change clocks for GPU 00000000:05:00.0.
```

So clocks cannot be fixed. They must instead be *sampled* and controlled by
protocol. Every block record carries per-GPU SM clock, temperature, power and
the throttle-reason mask.

### 0.1a Standing condition: GPU2 is thermally throttled in EVERY run

This has been true for every measurement in this project — there has never
been a run on this rig with an unthrottled 3080. The consequence is that all
arms and all historical anchors are **mutually comparable**, and no number
below needs to be discounted for it. What the numbers are is a **floor**: on
a cooler machine the same configuration lands higher.

Two standing caveats follow, and they are about *inference*, not validity:

* No placement / load-shifting recommendation is derived from the throttle
  state. "Card X is hot, therefore move work off it" is not a result — it is
  a property of this summer. Structural asymmetry (card class, memory
  bandwidth, head/layer geometry) is a valid basis; temporary throttling is
  not.
* Ratio recommendations produced under this condition — notably
  auto-performance's `--rank-mlp-ratio [6,1,1]`, MLP concentrated on the 5090
  — may partly encode the throttle effect. That is a caveat on their
  interpretation, not a reason to discard them.

Only the first ~1-2 points after a long idle are unthrottled; every sustained
run is throttled throughout. That is a **warm-up bias** that would hand a
free win to whichever arm is measured first, which is exactly why the
protocol below discards a fixed warm-up burn and records clock/temp per
point.

### 0.2 The rig thermally throttles, and the throttled card is a TP rank

Observed at the start of the campaign, mid-load:

| GPU | card | temp | SM clock | throttle |
|---|---|---|---|---|
| 0 | RTX 3080 | 83 C | 1920 MHz | none |
| 1 | RTX 5090 | 74 C | 2842 MHz | none |
| 2 | RTX 3080 | 88 C | 1620 MHz | **sw_thermal_slowdown ACTIVE** |

Two *identical* 3080s, 300 MHz / 18 % apart, purely thermal. Under lock-step
TP the slowest rank sets the pace, so this lands directly on end-to-end
tok/s. GPU2 is the system's pacing rank.

### 0.3 Warm-up transient: ~3 minutes, and it is large

A-vs-A series from cold (one unchanged config, 28 identical blocks):

| phase | GPU2 clock | GPU2 temp |
|---|---|---|
| blocks 0-8 (warm-up) | 1875 -> 1658 MHz | 75 -> 87 C |
| blocks 9-27 (plateau) | 1640 -> 1637 MHz | 88 -> 88 C |

**Protocol consequence:** every arm burns a 9-block (~3 min) warm-up that is
discarded before any measured point is taken. `arm_measure.sh` enforces it.

### 0.4 The detection limit, measured A-vs-A (not assumed)

Same config, same flags, plateau blocks only:

| class | n | median tok/s | sd | p5 / p95 | peak-to-peak |
|---|---|---|---|---|---|
| code | 19 | 90.60 | 2.31 (**2.56 %**) | 82.95 / 93.48 | 11.7 % |
| prosa | 8 | 68.64 | 1.74 (**2.55 %**) | — | 7.5 % |
| misch | 8 | 74.25 | 1.77 (**2.40 %**) | — | 7.5 % |

Paired adjacent differences (plateau, A vs A): median -0.44 %, sd 3.80 %,
p5/p95 -8.79 % / +9.48 %.

Resulting two-arm 95 % detection limits on **raw tok/s**:

| blocks per arm | detection limit |
|---|---|
| 6 | 4.10 % |
| 8 | **3.55 %** |
| 10 | 3.18 % |

This reproduces the #199 observation (80.45 / 91.52 / 89.75 = 13 % spread)
and confirms it was never an artefact: **a 1 % effect is not measurable on
this rig end-to-end.** The 5 % bar is only just resolvable at n=8.

### 0.5 The residual noise is CONTENT variance — and it factors out

In the plateau, `r(tok/s, spec_accept_length) = **0.9809**` (n=19). Every one
of 28 blocks produced a *different* output text despite greedy sampling
(temperature 0, identical prompt) — the known fp8/sm8x prefill
nondeterminism. Different text is differently predictable, so accept length
moves, and throughput follows it almost perfectly.

Speculative decode throughput factorises exactly:

```
tok/s  =  verify_rounds_per_second   x   accepted_tokens_per_verify_round
          ^ hardware / pipeline axis     ^ content / speculation axis
```

Dividing the measured accept length out of the measured tok/s isolates the
hardware axis, and the noise collapses:

| class | raw tok/s sd | accept sd | **round_rate sd** |
|---|---|---|---|
| code | 2.56 % | ~2.5 % | **0.55 %** |
| misch | 2.40 % | 2.17 % | **0.34 %** |

So the campaign reports **both** axes:

* `round_rate` — detection limit ~0.7 % at n=8. This is where a pipeline /
  placement / split change must show up.
* `accept` — the speculation-quality axis; this is where k must show up.
* `tok/s` — the product, i.e. what the user actually gets, with its honest
  3.5 % limit.

This is the "measure the mechanism, not only the end-to-end number" rule,
and here the mechanism split fell out of the data rather than needing a
profiler.

### 0.6 Protocol per arm

1. Boot with the arm's flags; record `max_total_num_tokens` (the KV capacity
   line) and per-rank pool/free memory.
2. Warm-up burn: 9 blocks, discarded.
3. Single session: 8 blocks x {code, prosa, misch}.
4. Dual session: 5 blocks x {code, prosa, misch}.
5. Every block records: tok/s, `spec_accept_length` (never the EMA), token
   count, output-text hash, per-GPU clock/temp/power/throttle.
6. Arms are interleaved against a re-measured baseline so that boot-to-boot
   and slow thermal drift are carried by paired differences, not absorbed
   into the effect.

### 0.7 Baseline capacity line

```
max_total_num_tokens = 98328   (uneven-DCP raw 4922176, clamped by the
                                hybrid mamba/attention cap:
                                max_running_requests=3 x (ctx 32768 + headroom))
available_gpu_mem = 10.11 GB
per-rank profiled capacity [2052169, 1538182, 1538186], active vector [6,5,5]
```

Because the raw uneven-DCP capacity (4.92 M tokens) exceeds the mamba cap
(98 328) by ~50x, the effective KV capacity is set by
`max_running_requests x context_len`, **not** by KV bytes. The "max-KV must
not fall > 10 %" gate is therefore insensitive to anything that only moves
KV bytes; it can only be tripped by a change that alters the cap itself or
drives raw capacity below ~98 k. Reported per arm regardless.

---

## 1. Vocab lead (#203) — `--rank-vocab-ratio`

Status: in measurement.

Flag facts established by code audit (not assumption):
* Parsed at `python/sglang/srt/server_args.py:432-437`; `auto` on this rig's
  measured membw (5090 1558 GB/s, 3080 723 GB/s) resolves to **`[13,6,6]`**
  via `uneven_perf.vocab_ratio_from_membw`. The requested `7,3,3` (rank 0 =
  53.8 %) is nearly the same operating point as auto (52.0 %).
* It shards `VocabParallelEmbedding` / `ParallelLMHead` on a 64-row padded
  grid, contiguous prefix-sum layout, so `index -> token_id` is preserved.
* **It does not touch KV accounting anywhere in code** (single consumer:
  `scheduler.py:4918`). It does change per-rank *weight residency*, which
  flows into the profiled KV budget indirectly.
* Two modelling gaps found, both worth reporting independently of the
  measurement outcome:
  1. `PerfCostModel` hardcodes the vocab family as evenly split
     (`uneven_perf.py:1702`), so auto-performance's predicted context is
     optimistic whenever a non-uniform vocab vector is active.
  2. `measured_kv_budget_fingerprint_fields` (`uneven_perf.py:507-556`) does
     **not** include `rank_vocab_ratio`. Under `SGLANG_MEASURED_KV_BUDGET=1`
     a budget measured without the vocab vector is silently reused with it.
     For this A/B that is harmless-to-helpful (it pins the budget identically
     across arms, which #188 wants), but it is the same defect class the code
     already documents for the MLP vector.

Mechanistic prior (to be falsified, not assumed): the lm_head GEMM is a small
part of a ~37.6 ms verify round on a memory-bound 27B decode, so the
reachable gain is small; and unequal logit shards **disable the equal-shard
multimem/NCCL gatherers** in `logits_processor.py`, which can make the arm
net-negative. Both effects live on the `round_rate` axis, where the detection
limit is ~0.7 %.

---

## 1a. Vocab lead (#203) — RESULT: does not clear the bar

Arm: `--rank-vocab-ratio 7,3,3` vs baseline, identical protocol otherwise.

Capacity gate: **PASSED exactly** — `max_total_num_tokens` 98328 in both arms
(0.00 % change). Weight residency shifts as expected (`available_gpu_mem`
10.11 -> 9.21 GB).

Single session (decode window 200->1600, baseline n=19 plateau, vocab n=8):

| class | metric | baseline | vocab | delta | det. limit |
|---|---|---|---|---|---|
| code | tok/s | 90.30 +- 2.31 | 95.19 +- 0.09 | +5.41 % | 1.15 % |
| code | accept | 3.388 +- 0.078 | 3.471 +- 0.000 | +2.43 % | 1.03 % |
| code | round_rate | 26.648 +- 0.146 | 27.425 +- 0.026 | +2.92 % | 0.25 % |

Dual session (n=5 per class), the axis that decides it:

| class | baseline round_rate | vocab round_rate | delta |
|---|---|---|---|
| code | 47.383 | 47.594 | +0.45 % |
| prosa | 46.798 | 46.745 | -0.11 % |
| misch | 46.381 | 46.735 | +0.76 % |

**Verdict: REJECTED for the working-arm recipe.** The bar is > 5 % single AND
dual. Dual is +0.45 / -0.11 / +0.76 %, i.e. flat and below the detection
limit on every class. The single-session number is real but does not
generalise to the loaded case, which is the mechanistically expected result:
at bs=2 the lm_head GEMM is amortised over more tokens per round and the
vocab imbalance stops mattering.

Two caveats recorded honestly rather than buried:
* The two arms sat at different pacing-rank clocks (GPU2 1629 vs 1711 MHz,
  +5.0 %). A fit of round_rate against GPU2 clock over the warm-up sweep
  (n=9, R2 = 0.909) gives slope 0.00102 round_rate/MHz — so a 12 % clock swing
  is worth only ~0.85 % throughput. Decode here is **VRAM-bandwidth bound and
  the memory clock is not throttled**, so the clock gap explains ~0.3 % of the
  +2.92 %, not all of it. The single-session gain is mostly structural.
* On the code prompt the vocab arm was **bit-deterministic** — one single
  output text across all 8 blocks (accept sd = 0.000), where the baseline
  produced 28 distinct texts. Interesting (unequal logit shards take a
  different, apparently deterministic reduction path) but it also means the
  arm's accept figure is ONE sample of the text distribution, not an estimate
  of its mean. Do not read the +2.43 % accept as an arm property.

## 2a. k-matrix — k=4 measured, RESULT: does not clear the bar

Protocol re-tightened to the timebox for this block: decode window 200->1000
(~13 s/point), 6 blocks single / 4 dual per class, 4-block warm-up burn.
Because the window changed, the baseline was **re-measured on a fresh boot
with the identical protocol** — the earlier 1600-token baseline is not
comparable and was not used here.

Capacity gate: `max_total_num_tokens` 98331 (k=4) vs 98328 (k=3) = **+0.003 %**.
The draft KV pool is essentially k-independent here; the 10 % KV limit is
nowhere near being touched.

Single session:

| class | k=3 tok/s | k=4 tok/s | delta | k=3 accept | k=4 accept | accept delta | k=3 rr | k=4 rr | rr delta |
|---|---|---|---|---|---|---|---|---|---|
| code | 91.61 | 94.83 | +3.51 % | 3.390 | 3.802 | +12.15 % | 27.026 | 24.941 | -7.71 % |
| prosa | 67.83 | 67.93 | +0.15 % | 2.622 | 2.811 | +7.21 % | 25.871 | 24.167 | -6.59 % |
| misch | 75.05 | 75.97 | +1.23 % | 2.884 | 3.164 | +9.71 % | 26.017 | 24.007 | -7.72 % |

Dual session, code: k=3 160.46 tok/s vs k=4 150.84 tok/s = **-6.00 %**
(accept +8.40 %, round_rate -13.32 %).

**Verdict: k=4 REJECTED.** The two axes move in opposite directions exactly as
the mechanism predicts — one extra draft step buys ~8-12 % more accepted
tokens per verify round but costs ~7-8 % of the round rate single and ~13 %
dual, because the draft forwards are serial and do not amortise. Net single
is +0.2 to +3.5 % (below the 5 % bar, and prosa/misch are below the detection
limit); net dual is **negative**. Under the "> 5 % single AND dual" rule this
fails on both counts.

The structure this exposes is the useful part: **the profitable direction for
k is workload-dependent through the round-rate term, and dual session pushes
the optimum DOWN, not up.** k < 3 was not measured (see gaps).

## Gaps — not measured, named rather than filled

* **k=1, k=2, k=5.** The k=4 result plus the round-rate mechanism predicts
  k=2 is the interesting direction for dual session, but it is unmeasured.
* **Split-balance (block 3).** Not measured. Mechanistically it is expected to
  be near-inert at the tested depth: the prompts run ~1.6 k context, where
  attention over owned KV tokens is a small share of a memory-bound decode
  round; the lever can only bite at depth. This is a prediction, not a result.
* **Dual prosa/misch for the fresh k=3 baseline** (run cut short).
* No policy thresholds were pinned, because no arm cleared the bar.

## Standing conclusion for the working arm

**No change recommended.** The working-arm recipe stays as it is
(NEXTN 3/1/4, TP=3 uneven DCP, flashinfer, CUDA graphs on). Two of this
task's three levers were measured and both fail the > 5 % single-AND-dual
bar; the third is unmeasured. Combined with #199 (collective overlap
cancelled) and #201, this is the third consecutive lever on this arm to come
back under the bar, which is itself a finding: **this configuration is close
to what the hardware allows, and the remaining headroom is not in the
speculation/split axis.**

---

# Task #204 — cross-rig link A/B (RDMA vs 1 GbE), measured 2026-07-26

Vehicle: Llama-3.1-8B fp16, cross-rig TP=4 (nnodes=4, one rank per node):
rank 0 = 5090, ranks 1/2 = 3080, rank 3 = 2080 Ti on the second rig. ctx 4096,
triton attention, no CUDA graph, no speculation (identical on every arm, so
the transport is the only variable). Raw: `/spinning/r3val/logs/link_*.json`,
`/spinning/r3val/logs/lat_*.json`.

**Decode rate is measured by streaming**, as `(n-1)/(t_last - t_first)`. This
was deliberate: the earlier L0 e2e-slope numbers are invalid because a
repeated identical prompt hits the radix prefix cache, so the constant prefill
term does not cancel out of the slope. Streaming sidesteps both prefill and
the cache.

Collective latency table and the end-to-end table are recorded in
`FEATURES_VS_UPSTREAM.md` under row 21 (HTCCL). Headline numbers:

* **RDMA vs 1 GbE, end to end: 3.6-3.7x** (28.44/28.56/27.13 vs
  7.78/7.72/7.54 tok/s over code/prosa/misch).
* **RDMA's advantage is latency, not bandwidth:** on the same wire it is 14x
  at barrier and 8x at 8 KiB, but only 1.2x at 4 MiB.
* **The "78 us" 1 GbE figure is superseded** — it was a raw TCP round-trip,
  not a collective. The 1 GbE collective barrier is 146.63 us.
* **Uneven TP does nothing here**, on either wire (-3.3 to +2.0 %).

## Why uneven TP is inert on this vehicle

Per-rank GPU utilisation sampled during decode (arm E, RDMA + uneven
3,2,2,1): **5090 10-13 %, 3080s 19-24 %, 2080 Ti 30-35 %.** Every rank is
mostly idle. The group is collective- and latency-bound, so shifting compute
between ranks has nothing to recover — and note the 2080 Ti is still the
busiest rank *after* being given only 1/8 of the heads, so the split is not
even close to equalising utilisation. Capacity cost of the uneven split:
`max_total_num_tokens` 140168 -> 129242 (-7.8 %).

The kv-aligned partitioner works as designed: `tp_partition_size` splits the
32 q heads in units keyed to the 8 kv heads, so ratio 3,2,2,1 gives kv
3/2/2/1 and q 12/8/8/4 with no ragged shard.

## Gap: arms A1/A2 (single-rig reference) — BLOCKED, not skipped

Intended as the no-rig-crossing reference at TP=3, ratio 3,3,2 (least skewed)
vs 4,2,2 (performance-weighted). **Both fail to boot**, in the container AND
on the PVE host:

```
flashinfer/norm/kernels/rmsnorm.py -> rmsnorm_cute
RuntimeError: CUDA Error: cudaErrorNoKernelImageForDevice
```

`flashinfer.norm.rmsnorm_cute` is JIT-compiled for ONE architecture and reused
on the other across the 5090 (sm120) + 3080 (sm86) group. Filed as **#208**.

Three hypotheses were tested and **falsified** — recorded so nobody repeats them:
1. *"container-only mixed-arch problem"* — no, it reproduces on the PVE host.
2. *"fp16 vs bf16 dtype"* — no, `--dtype bfloat16` fails identically.
3. *"`SGLANG_OPT_USE_JIT_NORM=0` bypasses it"* — no, fails identically.

Lead for #208: the cross-rig arms survive this on the same cards, and there
each rank has `CUDA_VISIBLE_DEVICES` pinned to a single GPU — so the
per-process JIT matches that process's one arch, and no cross-arch cache reuse
can occur. That points at a JIT cache key that omits the arch.

A separate, independent finding from the same attempt: **a strictly even TP=3
is not expressible for Llama-3.1-8B in this fork.** Plain TP=3 trips
`assert total_num_heads % tp_size == 0` (32 q / 8 kv are not divisible by 3),
and a uniform `--rank-tp-ratio 1,1,1` is rejected outright ("identical entries
is the even split — omit the flag instead"). Hence 3,3,2 as the least-skewed
reference rather than a true even split.
