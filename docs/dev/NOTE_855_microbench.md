# NOTE #855 — Marlin-wNa16 (W8A16) vs INT8-W8A8, measured

Step 0 of `ANALYSE_854_w8a16_vs_w8a8.md` §9 ("microbench before anything
else"), executed on metal in GPU window **W25**, 2026-08-24. Kernel
microbench only: no server, no checkpoint, random weights, ~15 min of card
time on all three cards.

**VERDICT (speed axis): the W8A16 path is VETOED ON SPEED.** It loses
2.9-3.3x against a GDN-covered W8A8 lane at prefill and 1.8x at decode on the
5090. What this bench settles is the *price*, not the *worth*: W8A16 has a
real, structural quality advantage that this bench does not measure and does
not refute (§3.5). The honest form of the result is an exchange rate — the
quality gain must be worth **2.05-2.27x prefill linear time and 1.8x decode
on the 5090** — and that trade is a decision, not a measurement.

The recommendation of ANALYSE_854 §9 step 3 — requant the incumbent ourselves
with GDN coverage, staying W8A8 — stands unchanged and is now quantified:
leaving the GDN dense projections in BF16 costs **1.39x (sm120) / 1.46x
(sm86)** of prefill linear-layer time today.

---

## 1. What was measured

| | |
|---|---|
| Arm A | `marlin_wna16` — `apply_gptq_marlin_linear`, `uint8b128`, `group_size=128`, weights through the serving path's own `gptq_marlin_repack` + `marlin_permute_scales`. This is the COMPLETE W8A16 op: the lane has no activation-quant step at all, which is its whole structural claim. |
| Arm B | `int8_fused` — `per_token_quant_int8` + `sgl_kernel.int8_scaled_mm`, i.e. the verbatim body of `CompressedTensorsW8A8Int8.apply_weights` (`compressed_tensors_w8a8_int8.py:213-217`), **including** the per-token activation quant. Comparing A against the bare GEMM (`int8_gemm`, also reported) would hand W8A16 a cost the deployed lane really pays. |
| Arm C | `bf16_linear` — `F.linear`, the floor reference and the lane the unquantized GDN projections run on today. |
| Shapes | ANALYSE_854 §4.1's worked 12:10:10 example, rank 0: `gate_up 13056x5120`, `down_proj 5120x6528`, `o_proj / GDN out_proj 5120x2304`. All K are 128-aligned because the wNa16 lane coarsens shards to 128, not to the INT8 lane's 16 — the two schemes do not share a shard table. |
| Points | decode `M=1` (eager **and** CUDA-graph replay) and prefill `M=2048` (eager). |
| Archs | sm120 (RTX 5090) and sm86 (RTX 3080), each pinned by **NVML UUID**. |
| Harness | `scripts/int8_368/microbench.py`, the #368 harness, extended with the `marlin_wna16` optional lane (opt-in, `--lanes ...,+marlin_wna16`), an ANALYSE_854 shape preset, and a CUDA-graph work-verification guard. #368 default invocations are unchanged and stay comparable. |

Raw JSON: `/spinning/gpu-arb/855-sm{120,86}-{decode,prefill}.json`.
Window record: `/spinning/gpu-arb/W25-RESULT-855-marlin-microbench.md`.

### 1.1 The #384 trap did not apply — checked, not assumed

Before the window, `kernel_dist_guard.py --require-arm
--expect-pinned-sha256` returned `verdict=ARMED`: single providing dist
`sglang-kernel 0.4.4`, `direct_url` pin
`67f03cfa755efa01498c7732bd6ae015ec5673feffe9a51452fefdbe0dcd4664`,
`int8_scaled_mm` resident in `sm100/common_ops.abi3.so`. Arm B is the real
deployed CUTLASS kernel, not a fallback and not the armless pypi wheel.

### 1.2 CUDA index != NVML index on this box (measured today)

`CUDA_VISIBLE_DEVICES=1` opens a **3080**; NVML/nvidia-smi index 1 is the
**5090**. Every run here is pinned by UUID.

    NVML 0 = RTX 3080  GPU-5c648f96-be1d-42d5-0221-34d11ab137f7   (CUDA idx 1)
    NVML 1 = RTX 5090  GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d   (CUDA idx 0)
    NVML 2 = RTX 3080  GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4   (CUDA idx 2)

Any ticket that pins a card by integer index on this rig pins the wrong card.

---

## 2. The table

Medians over 9 rounds of auto-calibrated bursts, interleaved lane rotation,
ms per op. `A/B` = Marlin / INT8-W8A8; `>1` means Marlin is slower.
`floor` = the A-vs-A spread between two independently built copies of the
same lane at that point (the noise floor), taken on the compared lanes.

### 2.1 Decode, M=1 — CUDA-graph replay (the deployment-relevant mode)

| arch | shape (N x K) | bf16 | int8_fused | marlin | **A/B** | floor |
|---|---|---:|---:|---:|---:|---:|
| sm120 | gate_up 13056x5120 | 0.08101 | 0.01326 | 0.02396 | **1.81** | 0.9 % |
| sm120 | down 5120x6528     | 0.04267 | 0.00872 | 0.01585 | **1.82** | 1.7 % |
| sm120 | out_proj 5120x2304 | 0.01644 | 0.00481 | 0.00881 | **1.83** | 0.7 % |
| sm86  | gate_up 13056x5120 | 0.20659 | 0.09748 | 0.09991 | 1.03 | 0.04 % |
| sm86  | down 5120x6528     | 0.09908 | 0.05421 | 0.05343 | 0.99 | 0.6 % |
| sm86  | out_proj 5120x2304 | 0.03807 | 0.02289 | 0.02196 | 0.96 | 1.7 % |

### 2.2 Decode, M=1 — eager (kept only to show the trap)

| arch | shape | int8_fused | marlin | A/B |
|---|---|---:|---:|---:|
| sm120 | gate_up | 0.04795 | 0.03318 | **0.69** |
| sm120 | down | 0.04746 | 0.03327 | **0.70** |
| sm120 | out_proj | 0.04593 | 0.03274 | **0.71** |
| sm86 | gate_up | 0.09993 | 0.10154 | 1.02 |
| sm86 | down | 0.05642 | 0.05490 | 0.97 |
| sm86 | out_proj | 0.04820 | 0.03266 | 0.68 |

### 2.3 Prefill, M=2048 — eager

| arch | shape | bf16 | int8_fused | marlin | **A/B** | marlin/bf16 | floor |
|---|---|---:|---:|---:|---:|---:|---:|
| sm120 | gate_up | 1.25222 | 0.46464 | 1.31861 | **2.84** | 1.05 | 0.5 % |
| sm120 | down | 0.65235 | 0.22472 | 0.65388 | **2.91** | 1.00 | 2.6 % |
| sm120 | out_proj | 0.22736 | 0.08618 | 0.24929 | **2.89** | 1.10 | 0.5 % |
| sm86 | gate_up | 4.68864 | 1.53373 | 5.45212 | **3.56** | 1.16 | 3.5 % |
| sm86 | down | 2.47455 | 0.81939 | 2.68186 | **3.27** | 1.08 | 3.2 % |
| sm86 | out_proj | 0.90423 | 0.31860 | 0.99133 | **3.11** | 1.10 | 2.4 % |

### 2.4 Noise floor, stated once

Compared-lane A-vs-A floors ranged 0.03-2.1 % at decode and 0.5-3.5 % at
prefill. Every prefill effect (211-256 % above parity) and every sm120
decode effect (81-83 %) stands 50-100x above its floor. The sm86 decode
effects (1.4-4.0 %) sit AT the floor — reported as a **wash**, not as a
result in either direction.

One outlier is recorded rather than smoothed: at sm120 `out_proj` the
`bf16_linear` A-vs-A floor was **43.9 %** (0.01676 vs 0.01165 ms between two
independently allocated weight copies) — a cuBLAS tile/alignment difference,
not clock noise. It touches only the BF16 reference at that one point; the
Marlin and INT8 floors there are 0.69 % and 1.02 %, and the §3 modelling uses
medians across all three shapes (the other two BF16 floors are 0.12-0.30 %).

---

## 3. What the numbers mean for the three open questions

### 3.1 Is W8A16 vetoed? YES — on prefill, on both cards

ANALYSE_854 §5.1 modelled prefill-GEMM time from a datasheet 2x INT8:1 BF16.
Measured on this rig, at M=2048, the INT8 lane is **2.7x (sm120) / 3.0x
(sm86)** faster than dense BF16, and Marlin is **1.05x / 1.10x SLOWER** than
dense BF16 (not the ~0.9x the analysis allowed). Rebuilding the §5.1 table
with measured coefficients, normalized to all-BF16 = 1.0, GEMM param split
77.2 % MLP+FA / 22.8 % GDN dense:

| checkpoint | INT8-computed share | sm120 | sm86 | (ANALYSE_854 model) |
|---|---:|---:|---:|---:|
| W8A8 + GDN coverage (§8.3, to be built) | 100 % | **0.371** | **0.331** | 0.50 |
| active INT8 (GDN in BF16) | 77.2 % | **0.514** | **0.484** | 0.61 |
| lued W8A16 (Marlin everywhere) | 0 % | **1.053** | **1.096** | 1.00 |

So W8A16 is **2.05x (sm120) / 2.27x (sm86)** slower than the checkpoint we
already run, and **2.84x / 3.31x** slower than the GDN-covered W8A8 we should
build. The analysis modelled 1.63x and 2.0x; measurement is worse than the
model on both, because the model under-credited INT8.

### 3.2 The decode unknown ANALYSE_854 §5.2 named — settled, and it is NOT the veto

The open question was "where Marlin W8A16 lands between int8 traffic and bf16
compute on sm86 at M=1..8". Answer: **on the INT8 lane, not between.** At
M=1 on sm86 both schemes are weight-traffic-bound and both read one byte per
weight, so they tie (0.96-1.03x, at the floor). Marlin/BF16 is 0.48-0.58x and
INT8/BF16 is 0.47-0.60x — the same halving from the same cause.

On sm120 the INT8 lane is fast enough that Marlin's dequant-plus-BF16-compute
becomes visible even at M=1: **1.81-1.83x slower**, far above the floor.

Against §9 step 0's own stop rule ("if Marlin comes out worse than 0.75x of
the W8A8 lane at decode M, stop here", i.e. slower than ~1.33x): the rule
**fires on sm120** (1.82x) and **does not fire on sm86** (0.99x). The veto
therefore does not rest on the decode axis — it rests on prefill, where both
cards agree decisively, plus the 5090's decode loss.

### 3.3 The #368 lesson reproduces — in the direction that would have misled us

At M=1 on sm120, **eager** says Marlin/INT8 = 0.69x (W8A16 wins) and **graph
replay** says 1.81x (W8A16 loses by 81 %). An eager-only bench would have
endorsed the W8A16 path at decode. The eager gap is the INT8 lane's
per-token activation-quant launch, which graph replay amortizes away —
exactly the constant #368 identified (commit `5df91f62fb`: `int8_quant`
0.0266 -> 0.0012 ms). The corollary is unchanged: **the activation-quant cost
that W8A16 structurally removes is not worth having.**

### 3.4 GDN requant in W8A8 (§9 step 3): needs no Marlin, and the BF16 floor is now priced

The recommendation stays on the INT8 lane and is untouched by this bench —
its role was only to settle W8A16 as a fallback candidate (it is not one) and
to price the BF16-GDN cost floor. That price, from the §3.1 table:

**Keeping the GDN dense projections in BF16 costs 1.39x (sm120) / 1.46x
(sm86) of prefill linear-layer time** (0.514 -> 0.371 and 0.484 -> 0.331).
That is the recurring cost of the incumbent producer's `re:.*linear_attn.*`
exclusion, paid on every cold-context prefill, on top of the 4.87 GiB /
+156k-token KV win the same exclusion also costs us. Both halves of the case
for requanting ourselves are now measured rather than modelled.

### 3.5 The quality axis is NOT settled by this bench — and it favours W8A16

A speed bench cannot veto a quality argument, and the quality argument for
W8A16 is structural, not a Reddit chart:

1. **W8A16 quantizes no activations at all.** The W8A8 lane runs
   `per_token_quant_int8` on every activation tensor before every linear
   (`compressed_tensors_w8a8_int8.py:213`) — dynamic per-token INT8, i.e. 8
   bits for a distribution with outlier channels. In published W8A8-vs-W8A16
   comparisons the *activation* side, not the weight side, is usually the
   dominant error term. W8A16 pays zero of it: activations stay BF16 into
   Marlin's BF16 tensor cores.
2. **W8A16's weight quantization here is also finer**: group 128 along K
   (one scale per 128 weights) versus W8A8's per-channel (one scale per
   output row). More scales, less per-group range to cover.

So "BF16 activations should simply be numerically better than INT8
activations" is correct as stated, and nothing measured above contradicts it.
This bench deliberately measured the axis that could be settled cheaply
(ANALYSE_854 §9 step 0, "cheapest falsifier first"); the quality axis is §9
step 2 and needs the KLD/quality suite **with a W8A8 arm, measured through
the Marlin serving path** — precisely the comparison the lued card does not
contain (it measures W8A16 vs FP8, on the HF decompressed path).

What the measurement does change about that decision:

- The price is **higher than ANALYSE_854 modelled** (2.05-2.27x vs the
  modelled 1.63x against the active checkpoint), so the quality bar W8A16
  must clear is correspondingly higher.
- The penalty is **asymmetric across the rig and across phases**: decode on
  the two 3080s is a wash (0.96-1.03x), decode on the 5090 is 1.81x, prefill
  is 2.8-3.6x everywhere. A quality-first profile is therefore not absurd —
  it is a prefill-throughput trade, mitigated to the extent HiCache/radix
  prefixes hit and cold context is rare.
- If quality is the goal, **lued is still not the candidate to buy it with**
  (ANALYSE_854 §9 step 4): a calibrated per-channel AutoRound W8A16, or our
  own `scheme: W8A16, strategy: channel` requant, beats data-free RTN on
  method and drops the scale tax from 362.5 MiB to ~7.4 MiB. Its Marlin cost
  was not measured here (§4) but its dominant term is the same BF16 compute,
  so expect the same order of penalty.
- **The two axes are separable.** GDN coverage (the 4.87 GiB / +156k-token
  KV win) is a *recipe* property available on the W8A8 lane at no speed cost
  — in fact at a 1.39-1.46x prefill *gain*. Nothing about wanting better
  quality requires taking the W8A16 speed penalty to get the VRAM win.

**Restated verdict:** W8A16 is vetoed as a *drop-in performance-neutral
alternative* and as the *W8A8 fallback candidate* — it is neither. It is not
vetoed as a deliberate quality-for-throughput trade; that decision needs §9
step 2, and this note supplies the price side of it.

---

## 4. Coverage and honesty notes

- **Deliberately small matrix** (user directive 2026-08-24: the minimum
  decision set, not a battery). Measured: M ∈ {1, 2048} on 3 shapes on 2
  archs. NOT measured: M = 2, 4, 8, 512; the other 7 shapes of the #855 set
  (ranks 1/2 shards and the un-sharded 27B projections); the third card
  (second 3080, same arch as the first).
- **Prefill was measured eager only.** At M=2048 an op costs 0.09-5.5 ms
  while a launch constant is ~25 us, so graph replay cannot move a 3x effect
  — but this is an argument, not a measurement, and is named as such.
- **`bf16_linear` is `F.linear`/cuBLAS**, the same call the unquantized GDN
  projections make today. It is the deployed BF16 lane, not a tuned kernel.
- **group_size = 128 only.** ANALYSE_854 §9 step 4's preferred W8A16 variant
  (per-channel, `group_size = -1`, e.g. Minachist INT8-AutoRound) was NOT
  measured. Its scale traffic is smaller, but the dominant term here is BF16
  tensor-core compute, which per-channel scaling does not change; expecting
  it to reverse a 2.8-3.3x prefill gap would need its own measurement.
- **Random int8 weights.** Neither Marlin nor CUTLASS-int8 has a
  value-dependent fast path, so values do not affect the timings.
- **CPU-sampled inputs moved to device** (CUDA-randn cross-arch rule), so
  both cards see identical input bytes.
- **Empty-graph falsifier added** (#591 canon): every capture now zeroes its
  output, replays, and requires a non-zero result before it may be timed.
  No capture failed and no lane was skipped in any of the four runs — so the
  graph numbers above are of graphs that demonstrably do work.

## 5. Harness changes (this commit)

`scripts/int8_368/microbench.py`, additive only:

- `marlin_wna16` in `OPTIONAL_LANES` — opt-in, so every #368 default
  invocation keeps its exact lane set and stays comparable to recorded #368
  results.
- Marlin weight construction in `build_weights` through the serving path's
  own helpers, with Marlin's shape rules (K % 128, N % 64) checked in Python
  and reported as a note instead of crashed in CUDA.
- `--shape-preset {855,855min}` + `--drop-derived-shapes`: measure a literal
  ANALYSE_854 shape set without deriving an INT8 shard plan (and without
  needing a checkpoint on disk — the default config path points at
  Qwen3.6-27B-INT8-W8A8, which is no longer the standard model).
- `capture_graph` empty-capture falsifier (see §4).
- `marlin_over_{int8_fused,bf16_linear,int8_gemm}[@graph]` in `derived`.
