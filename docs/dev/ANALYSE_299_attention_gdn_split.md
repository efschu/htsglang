# Task #299 — pulling the attention/GDN family into the phase split

CPU-only investigation, 2026-07-31. Branch `analysis/299-attn-gdn-split`, base
`4970939962`. No card window was taken (reasons in §8).

**Verdict up front.** The briefing's premise is factually wrong, and once
corrected the feature it proposes is worth less than the measurement noise
floor.

1. Attention and GDN are **not** uniformly split. On this rig they sit at
   `[0.50, 0.25, 0.25]` — *more* concentrated on the 5090 than the MLP family
   is (`[0.463, 0.272, 0.265]`). What they lack is an *own* family vector; they
   follow the base `--rank-tp-ratio`.
2. Both families have the **same per-rank speed ratio** (MLP 6.23x/6.64x,
   attention+GDN 6.53x/6.41x against the 5090). There is no comparative
   advantage to exploit, so a second, independent vector adds **0.16 ms of a
   245 ms compute optimum = 0.01 % of the prefill window**.
3. Under the real binding constraint (rank-0 VRAM) the gain rises to
   **+0.4 % … +1.8 %** of prefill across every physically valid parameter
   corner — still under the s=8 noise floor of **3.18 %**.
4. The clone variant is rejected on six independent grounds (§6), the first of
   which is fatal on its own: a clone is real bytes on the 5090, and rank-0
   VRAM is precisely the constraint that limits the existing, already-measured
   MLP lever.
5. **The actual money found by this investigation is somewhere else**: the
   `-84 %` context collapse that #296 charged against the prefill optimum
   `10,1,1` is an artefact of leaving `--rank-kv-ratio` at `7,3,3`. With a
   capacity-matched KV vector the same boot carries **6.18x** the tokens. One
   flag, no code. See §7.

---

## 1. Where the time is — the measured prefill cost model

### 1.1 Source data

`/spinning/gpu-battery-results/2026-07-30_phasen_optima/s15_phasen_optima/wait/`
(`anchor.json`, `prefill_opt.json`), medians over 15 prefill chunks of 2048
new tokens at 8 sessions, CollectiveClock-instrumented. Both arms share one
base plan (`--rank-tp-ratio auto-performance` -> VRAM-auto weights
`28107,16280,16280`) and differ only in `--rank-mlp-ratio`.

| arm | `--rank-mlp-ratio` | MLP units | TP0 comp | TP1 comp | TP2 comp | window |
|---|---|---|---:|---:|---:|---:|
| 1 anchor | (none, base) | 63, 37, 36 | 153.4 | 536.1 | 544.4 | 1519.2 |
| 2 prefill opt | `10,1,1` | 113, 11, 12 | 224.9 | 309.1 | 311.8 | 1311.8 |

TP0 = RTX 5090, TP1/TP2 = RTX 3080 20 GB.

### 1.2 The fit

Two arms, one axis moved, so the per-rank cost decomposes exactly. With
`m_r` = that rank's share of the dense-MLP family and `n_r` the non-MLP
residual:

```
comp_r = a_r * m_r + n_r
a   = [ 193.2, 1202.8, 1282.4 ]  ms   cost of 100 % of the MLP family on rank r
n   = [  64.0,  208.9,  205.0 ]  ms   this rank's actual non-MLP share
```

The fit reproduces all six measured compute values to 0.1 ms.

**Physical cross-check.** Rank 1's MLP segment is `1202.8 * 0.272 = 327 ms` for
`2 * 4.65e9 * 2048 = 1.905e13` FLOP = **58.3 TFLOPS**, against the #213 card
probe's `gemm_bf16_tflops = 65.57` for this card (89 % efficiency). Rank 0's is
`193.2 * 0.4632 = 89.5 ms` for `3.246e13` FLOP = **363 TFLOPS** against
`gemm_fp8_tflops = 566.88` (64 %). The model is not a curve fit, it lands on
the hardware.

### 1.3 The lockstep window identity

`window = max_r(comp_r) + collective_floor`, the fork's own model
(`INTEGRATION_R3_VALIDATION.md:4768`, #264: "window -7.6 % — almost exactly the
max-compute drop, as the lockstep model window = max_compute + collective_floor
demands").

```
floor(anchor) = 1519.2 - 544.4 = 974.8 ms
floor(arm 2)  = 1311.8 - 311.8 = 1000.0 ms
```

The floor is stable to 2.6 % and is **64-76 % of the whole window**. Everything
below is about the remaining quarter.

### 1.4 The "390 ms" of #252, and how much of it is left

#252 measured the 5090 waiting ~390 ms longer than the 3080s at the auto split.
In this data set:

| arm | TP0 wait | TP1 wait | 5090's excess wait |
|---|---:|---:|---:|
| 1 anchor | 1366.0 | 983.0 | **383.0 ms** |
| 2 prefill opt `10,1,1` | 1086.9 | 1001.6 | **85.3 ms** |

**The MLP lever has already harvested 78 % of the #252 imbalance.** The
residual that #299 targets is 85 ms, not 390 ms. That single row is the
strongest argument in this document and it should be read before anything
else.

---

## 2. The premise is wrong: attention/GDN are already ratio-split

Executed against the fork's own partition code
(`python/sglang/srt/distributed/utils.py`) with the anchor's base weights:

```
set_tp_partition_ratios([28107, 16280, 16280]);  tp_size = 3

MLP units (136)         -> [63, 37, 36]     <- reproduces the boot log exactly
GDN k-head units (16)   -> [ 8,  4,  4]
full-attn kv heads (4)  -> [ 2,  1,  1]
q heads (24)            -> [12,  6,  6]
vocab                   -> EVEN by default (only --rank-vocab-ratio opts in)
```

So the actual family shares on every #296 arm were:

| family | rank 0 | rank 1 | rank 2 | driven by |
|---|---:|---:|---:|---|
| dense MLP | 0.463 | 0.272 | 0.265 | `--rank-mlp-ratio`, else base |
| GDN | **0.500** | 0.250 | 0.250 | base `--rank-tp-ratio` only |
| full attention (q/k/v/o) | **0.500** | 0.250 | 0.250 | base `--rank-tp-ratio` only |
| vocab / lm_head | 0.333 | 0.333 | 0.333 | even unless `--rank-vocab-ratio` |
| KV tokens (DCP) | 7/13 | 3/13 | 3/13 | `--rank-kv-ratio` |

The 5090 already carries **half** of the attention and GDN work. The
"chronically underloaded 5090" observation is real, but it is an MLP-axis
statement, not an attention-axis one.

Correcting `n_r` for the real shares gives the attention+GDN family coefficient:

```
b_r = n_r / share_r
b   = [ 128.0, 835.6, 820.0 ] ms   cost of 100 % of attention+GDN on rank r
```

### 2.1 No comparative advantage — the decisive number

```
5090 advantage, MLP           :  a1/a0 = 6.23x   a2/a0 = 6.64x
5090 advantage, attention+GDN :  b1/b0 = 6.53x   b2/b0 = 6.41x
```

The two families rank the cards **identically**. Ricardo does not apply: there
is no family that "belongs" on the 5090 more than the other. Consequently the
two optima coincide:

```
optimum with the MLP axis alone (attention pinned at 8,4,4 / 2,1,1):
    max comp = 245.29 ms   at m = [0.938, 0.030, 0.031]  (MLP units 128, 4, 4)

optimum with BOTH axes free:
    max comp = 245.13 ms

difference = 0.16 ms  =  0.01 % of the prefill window
```

**An attention/GDN family vector adds one part in ten thousand to what the MLP
vector can already reach, in the absence of any VRAM constraint.** Everything
that follows is about whether the VRAM constraint changes that.

---

## 3. Splittability matrix

Code positions from a full read of the fork's partition plumbing.

| family | split dimension | granularity on this model | own family vector? | representable today? | hard blocker? |
|---|---|---|---|---|---|
| dense MLP | `intermediate_size` in 16/quant-block units | **136 units** (17408/128) | **yes** — `"mlp"` | yes | none |
| MoE experts | expert index | n/a here | **yes** — `"moe"` | yes | none |
| vocab / lm_head | vocab rows in padded units | 3880 units | **yes** — `"vocab"` | yes | solved (pad-to-max gather) |
| **full attention** | kv heads, q heads follow in kv-group packets | **4 kv heads** -> `[2,1,1]`, q `[12,6,6]` | **no** — base plan only | only via base `--rank-tp-ratio` | two geometric ones, see below |
| **GDN / gated delta net** | `num_k_heads` as unit family (`gdn_tp_units`) | **16 units** | **no** — base plan only | only via base `--rank-tp-ratio` | none structural; state-pool cost |
| KV tokens (DCP) | token ownership | continuous | **yes** — `--rank-kv-ratio` | yes | none |

### 3.1 Code positions

**The family dispatch, and the exact line where attention diverges**

* `python/sglang/srt/distributed/utils.py:834` `tp_partition_sizes(total, tp_size, units, family, groups)` — `ratios = get_tp_partition_ratios(family)`.
* Registered families: `python/sglang/srt/managers/scheduler.py:5367-5393` — exactly `("mlp", "moe", "vocab")`, installed via `set_tp_partition_ratios(base_plan, families=...)`.
* **The divergence line**: `python/sglang/srt/configs/model_config.py:1313` `_uneven_tp_num_kv_heads` calls `partition_sizes(..., weights=get_tp_partition_ratios())` — **bare, no `family=` argument**, where MLP passes `family="mlp"`. That single call site is why attention cannot be steered independently.
* Q heads: `python/sglang/srt/distributed/utils.py:1045` `attn_q_partition_units`, `:1084` `attn_q_partition_groups`.
* GDN: `python/sglang/srt/models/qwen3_5.py:209-226` — `gdn_tp_units = _quant_block_aligned_units(value_dim, num_k_heads, ...)`, then `tp_partition_size(num_k_heads|num_v_heads, ..., gdn_tp_units)`. Sibling `python/sglang/srt/models/qwen3_next.py:130-135`. **No `tp_family=` anywhere in the GDN stack** — a grep for `tp_family` yields only `"mlp"` and `"moe"`.
* MLP unit coarsening (`a6f7192c20`): `python/sglang/srt/models/llama.py:102`, `python/sglang/srt/models/qwen2_moe.py:208,222-224`.
* Flashinfer per-rank head counts (`f7ff514358`): `python/sglang/srt/layers/attention/flashinfer_backend.py:~316-360` `_local_attn_head_counts`.
* KV token vector: `python/sglang/srt/server_args.py:2039-2081`, resolution `python/sglang/srt/distributed/utils.py:420` `resolve_cp_token_ratios`.
* Vocab: `python/sglang/srt/distributed/utils.py:880` `tp_vocab_ratios` — deliberately does **not** fall back to the base plan ("vocab always even by design, M22").

### 3.2 Hard limits vs. "not wired up"

**Attention — collectives are not a blocker.** `o_proj` is `RowParallelLinear`
(`python/sglang/srt/layers/linear.py:2080`, all-reduce at `:2126`); the operand
is always full `hidden_size`, so the collective is shape-invariant under any
head split (`DESIGN_201:1185-1186`). The DCP head all-gather already has an
uneven variant via pad-to-max
(`python/sglang/srt/layers/dcp/comm.py:147` `cp_all_gather_heads_uneven`),
explicitly documented CUDA-graph-safe because the counts are boot-static.

Two genuine geometric blockers, both already adjudicated:

* **`kv_heads == tp_size`**: only the even q split is non-straddling; the #105 ragged kernel rejects anything else. The `<` -> `<=` flip was tried, measured and reverted (`INTEGRATION_R3_VALIDATION.md:534-599`); the real fix needs "a ragged kernel that supports per-rank non-uniform GQA mapping (the #169 head-gather family)".
* **uneven-TP MLA**: hard reject, `python/sglang/srt/layers/dcp/comm.py:38` `_reject_uneven_tp_mla`.

Plus model coverage: 95 of 102 attention-output sites still lack `tp_units`;
three models hard-reject a non-uniform plan rather than mis-shard
(`INTEGRATION_R3_VALIDATION.md:385-447`).

Neither blocker binds on Qwen3.6-27B (kv=4 > tp=3), so **for this model the
attention family vector is "not wired up", not "impossible"**. The work is:
thread `family="attn"` through `model_config.py:1313`, the model q/kv/o call
sites, `_local_attn_head_counts`, `cp_local_head_bounds`, and KV-pool sizing in
`pool_configurator.py`.

**GDN — no structural blocker at all**, purely not wired. The stack already
partitions on an arbitrary weight vector with `gdn_tp_units`, `out_proj` is
`reduce_results=False` RowParallel (shape-invariant), and the state cache
derives its per-rank shape from the same units
(`python/sglang/srt/configs/qwen3_next.py:346-366`). Adding `tp_family="gdn"`
is a mechanical edit of ~20 call sites. The reason it does not exist is a
**cost** argument, recorded twice:

* `python/sglang/srt/uneven_perf.py:31-33` — "SSM/GDN shifting is deliberately NOT a lever: the mamba state pool moves with the GDN units (~4.7 MiB/req/unit) and collapses context (M22 C3)."
* `docs/DESIGN_201_hierarchische_parallelitaet.md:1198-1200` — "GDN: mixer INCLUDING the qkvz projections stays FIXED (the state sticks to the rank; a move of ~19 MiB/request per tick is lethal) — switchable is ONLY the FFN/MLP part of each layer."

§4 tests whether that cost argument still holds numerically. It does.

### 3.3 The base vector already *is* an attention/GDN family vector

Worth stating explicitly for follow-up agents: because `--rank-mlp-ratio` and
`--rank-vocab-ratio` override their families, **the base `--rank-tp-ratio`
already behaves as a de-facto attention+GDN vector today**. A boot with

```
--rank-tp-ratio 10,3,3  --rank-mlp-ratio 10,1,1  --rank-vocab-ratio ...
```

steers attention/GDN to `[10,3,3]` while MLP stays at the prefill optimum. No
code is needed to *measure* the axis — only to name it cleanly. §5 uses this.

---

## 4. What the axis is worth once VRAM is in the model

### 4.1 The VRAM model, calibrated on the boot logs

All constants measured, none assumed:

| quantity | value | source |
|---|---|---|
| KV cost | 32.0 KiB/token, **identical on every rank** | `proofs/anchor.txt:275-284`; kv heads are replicated under DCP, tokens are the split |
| MLP unit | 3912 tokens of rank-local KV capacity | 195,587 tokens over 50 units, anchor -> arm 2 |
| GDN unit weight | 331.5 MiB (5304 MiB / 16 units) | model geometry, fp8 |
| GDN unit state | ~75 MiB (4.7 MiB/req/unit x 16 running requests) | `uneven_perf.py:31-33` |
| full-attn kv unit | 400 MiB (1600 MiB / 4 units) | model geometry, fp8 |
| anchor rank capacities | 233163, 108054, 112596 tokens | `proofs/anchor.txt:269-271` |

Model geometry (`config.json`, Qwen3.6-27B-FP8): hidden 5120, 64 layers
(48 linear_attention + 16 full_attention), `intermediate_size` 17408,
`num_attention_heads` 24, `num_key_value_heads` 4, `head_dim` 256,
`linear_num_key_heads` 16, `linear_num_value_heads` 48, vocab 248320,
`attn_output_gate: true`. Weight budget at fp8: MLP 16320 MiB (60.7 %),
GDN 5304 MiB (19.7 %), full attention 1600 MiB (5.9 %), embed + lm_head
2425 MiB (9.0 %).

Validation: the model reproduces arm 2's max compute as 318.2 ms against
311.8 measured (+2.1 %) and its prefill as +17.5 % against +14.2 % measured.
**The model is optimistic by ~3 percentage points on absolute prefill**, so
absolute predictions below are upper bounds; *differences* between two points
computed the same way are the trustworthy output.

### 4.2 Constrained search

Integer search over all `(MLP units, GDN units, kv heads)` allocations with
every rank's KV capacity held above a floor, minimising `max_r comp_r`:

```
BEST, attention pinned at today's 8,4,4 / 2,1,1 :  max comp 280.4 ms
BEST, attention family vector free              :  max comp 275.9 ms
  -> 4.5 ms of a 1255 ms window  =  +0.36 % prefill
```

Sensitivity across every corner of the two parameters that are estimated rather
than measured — `gamma` (GDN's share of the attention family's *compute*, best
estimate 0.60 from the FLOP split) and the GDN state surcharge — plus two KV
floors:

| gamma | GDN state MiB/unit | KV floor | pinned | free | prefill gain |
|---:|---:|---:|---:|---:|---:|
| 0.45 | 0 | 8192 | 280.4 | 279.4 | +0.08 % |
| 0.45 | 0 | 20000 | 297.3 | 288.8 | +0.67 % |
| 0.45 | 75.2 | 8192 | 280.4 | 270.0 | +0.83 % |
| 0.45 | 75.2 | 20000 | 297.3 | 282.4 | +1.17 % |
| 0.60 | 0 | 8192 | 280.4 | 263.6 | +1.34 % |
| 0.60 | 0 | 20000 | 297.3 | 274.1 | +1.83 % |
| **0.60** | **75.2** | **8192** | 280.4 | 275.9 | **+0.36 %** |
| **0.60** | **75.2** | **20000** | 297.3 | 291.8 | **+0.44 %** |
| 0.75 | 0 | 8192 | 280.4 | 246.3 | +2.72 % |
| 0.75 | 0 | 20000 | 297.3 | 250.6 | +3.68 % |
| 0.75 | 75.2 | 8192 | 280.4 | 259.4 | +1.67 % |
| 0.75 | 75.2 | 20000 | 297.3 | 268.8 | +2.24 % |

Bold rows are the best-estimate parameters. **The s=8 prefill noise floor is
3.18 %.** Exactly one of twelve corners clears it, and it requires
`GDN state = 0` — physically false, the state pool provably moves with the
units. In every valid corner the gain is **+0.4 % … +2.2 %**, i.e. not
measurable on this rig.

### 4.3 The user's direction specifically: 5090 takes *more* attention/GDN

Holding MLP at the prefill optimum `10,1,1` and pushing GDN units onto rank 0:

| GDN units | max comp | prefill vs anchor | vs arm 2 | rank-0 KV capacity |
|---|---:|---:|---:|---:|
| `8,4,4` (today) | 318.2 | +17.5 % | — | 37,563 tok |
| `10,3,3` | 287.4 | +20.4 % | **+2.4 %** | 11,534 tok |
| `12,2,2` | 256.7 | +23.4 % | +5.0 % | **-14,494 tok — unbootable** |
| `14,1,1` | 253.3 | +23.7 % | +5.3 % | **-40,523 tok — unbootable** |

**Exactly one step in the proposed direction is physically bootable, and it is
worth +2.4 % — inside the 3.18 % floor.** The second step does not fit on the
card. The compute-side ceiling of the whole idea (+5.3 %) is unreachable for a
VRAM reason that no amount of scheduling cleverness removes.

---

## 5. Why the exchange rate looked promising and still fails

Per MiB of rank-0 VRAM spent, the attention family *is* the better buy:

```
                 3080 relief      5090 load
dense MLP        0.0691 ms/MiB    0.0111 ms/MiB
attention+GDN    0.1210 ms/MiB    0.0185 ms/MiB     <- 1.75x more relief per MiB
```

Attention+GDN carries 1.64x more compute per weight byte than MLP (GDN's
conv1d, chunked delta rule, state update and norms are work without weights).
That is the one real argument for #299 and it is why the naive VRAM-free
optimum and the VRAM-constrained optimum disagree.

It fails for two reasons that compound:

1. **The GDN state surcharge eats the advantage.** A GDN unit costs 331.5 MiB
   of weight *plus* ~75 MiB of state pool at 16 running requests — a 23 %
   surcharge that the MLP unit does not pay. Comparing the 0.60/0-state row
   (+1.34 %) with the 0.60/75.2 row (+0.36 %) isolates it: **the state pool
   removes three quarters of the gain.** `DESIGN_201:1198-1200` reached this
   conclusion qualitatively; this is the number behind it.
2. **The MLP axis is not yet exhausted where it matters.** The unconstrained
   MLP optimum wants 128 units on rank 0; arm 2 booted 113. The remaining MLP
   headroom is cheaper per ms than the first attention unit, so the attention
   axis only starts paying after the MLP axis has already run into the wall —
   and the wall is at ~120 units, past which rank 0 has no KV pool left.

---

## 6. Clone vs. static re-split — verdict

**Static re-split**: representable today via the base vector (§3.3), worth
+0.4 % … +2.2 %, below the floor. Not worth building as a named flag.

**Dynamic clones** (rank 0 holds a copy of another rank's attention/GDN shard
and pulls work when idle): **reject.** Six independent grounds, any one
sufficient:

1. **A clone is real bytes, competing for the exact scarce resource.**
   `#93` is *physical VMM remap*, not N-virtual-aliases-over-one-physical —
   stated outright at
   `python/sglang/srt/speculative/adaptive_graph_memory.py:33-36`. There is no
   free-clone primitive. And rank-0 VRAM is the constraint that caps the
   already-measured +14.2 % MLP lever (§4.3). A clone spends the budget that
   produces the gain it is trying to add to.
2. **Swap latency exceeds the round.** Measured #93 remap on this rig:
   **40-51 ms organic, 85 ms max for ~1 GB**
   (`adaptive_graph_memory.py:207-214`) against a ~25-30 ms decode round. The
   offload register's own physics note already says it: "a 1.8-GB drafter per
   ~25-30 ms round is NOT hideable" (`offload_register.py:63-77`).
3. **The lending contract forbids this borrower class.**
   `python/sglang/srt/model_executor/dual_group_lane.py:3868-3877`: only
   *evacuable* content may occupy lent bytes; permanent posts are refused. A
   weight clone is a permanent post.
4. **GDN state sticks to the rank.** ~19 MiB/request/tick to move
   (`DESIGN_201:1198-1200`). This was adjudicated for the phase-dual MLP split
   and applies unchanged.
5. **Graphs forbid per-round work-volume changes.** Only a pre-captured ladder
   is legal (`DESIGN_201:1691-1699`;
   `kv_pressure_ladder.py:47-52`; the three symmetry assertions at
   `model_runner.py:1194`, `base_runner.py:63-94`,
   `decode_cuda_graph_runner.py:1298-1302`). HTCCL has **no `all_gatherv`**
   (`parallel_state.py:1732`) and its uneven `all_to_all` is explicitly not
   capturable (`htccl.py:936-944`); BAR1 round plans are baked per captured
   shape (`htccl_bar1.py:2798-2801`). Cost unit = graph pool x rungs.
6. **No sensor exists.** `CollectiveClock`
   (`python/sglang/srt/utils/collective_clock.py:69`) is armed for *plain
   prefill forwards only* and is blind under graph replay
   (`:38-45`). `LaneShareMeter._lane_rung()` returns the constant `"static"`
   (`scheduler.py:4703`). The decode rounds a dynamic controller would target
   are not instrumented at all.

**The arithmetic closes the case independently of the mechanism**: the *static*
ceiling of a perfect, cost-free, instantaneous re-split is +0.4 % … +2.2 %. A
dynamic version pays switching cost on top and therefore lands below zero.
This is the same shape as the #264 verdict ("thesis confirmed as diagnosis,
refuted as lever").

### 6.1 Second uses, named

The two primitives the proposal would have needed are worth building *for other
reasons*, and that is where the effort should go if it goes anywhere:

* **Peer-VRAM parking in the offload register (#286 GPU phase).** Everything is
  code except the P2P probe that fills `PeerPathCapability`
  (`offload_movement.py:36-49`, all numbers placeholders) and
  `DEFAULT_PARK_TARGET_ORDER = ("host_ram",)` at `:118`. Its real customer is
  KV session spill and the #274 lanes, not attention clones.
* **Decode-side CollectiveClock.** The prefill-only, graph-blind limitation
  blocks *every* dynamic controller, including the #274 lane ladder whose
  `_lane_rung()` is still hardcoded. This is the prerequisite for anything
  adaptive, and it is independent of #299.

---

## 7. The finding that is actually worth acting on

`--rank-kv-ratio` was left at `7,3,3` on **every** #296 arm while the weight
split moved underneath it. Because `max_total_num_tokens = sum(v) * min_r(cap_r
/ v_r)`, a KV vector that does not track the per-rank capacities throws away
context on the ranks that have it.

Recomputed against each arm's own measured per-rank capacities:

| arm | measured per-rank capacity | vector used | got | capacity-matched vector | would get | factor |
|---|---|---|---:|---|---:|---:|
| 1 anchor | 233163, 108054, 112596 | `7,3,3` | 433,017 | `13,6,6` | 448,390 | 1.04x |
| **2 prefill opt `10,1,1`** | 37576, 206358, 197718 | `7,3,3` | **69,784** | `2,11,10` | **431,475** | **6.18x** |
| 3 decode opt `7,3,3` | 195720, 133590, 133590 | `7,3,3` | 363,480 | `10,7,7` | 458,022 | 1.26x |

The sum of per-rank capacities is approximately **conserved** under a weight
re-split — moving 1 MiB of weight off rank 1 onto rank 0 costs rank 0 32 tokens
and gives rank 1 32 tokens. Measured: 453,813 (anchor) vs 441,652 (arm 2), a
2.7 % loss from prefill scratch scaling with the MLP shard, not the 84 % that
was charged.

The boot log already prints the fix as an unused hint —
`proofs/anchor.txt:268`: *"Uneven DCP: restart with
SGLANG_UNEVEN_TOKEN_VECTOR=33,15,16 to raise max_total_num_tokens from 433017
to ~450368 … active vector [7, 3, 3] leaves ranks idle."*

**Consequence for #296's verdict.** The prefill optimum was judged as
"+14.2 % prefill for +11.1 % ms/Verify **and -84 % context**". Two of those
three survive; the third is a flag setting. That materially improves the
economics of the phase-dual ladder (#274/#287) and should be re-stated before
the ladder is costed again.

**Caveat, honestly.** A `2,11,10` KV vector puts 21 of 23 token-shares on the
3080s, which moves decode attention *token* work onto the slow cards and
lengthens the DCP hop. Arm 4 measured the mild version of this (`1,1,1` vs
`7,3,3` at MLP `7,3,3`): bs=1 **-3.0 %** ms/Verify, bs=8 -0.4 % — i.e. mildly
*positive*. The extreme version is unmeasured and is the thing the follow-up
boot must check.

---

## 8. Recommendation

### 8.1 Do not build the attention/GDN family vector

Register as **discarded**, with the reason and the number, so it is not
retried without new information:

> #299 — attention/GDN family vector. Attention and GDN already follow the base
> `--rank-tp-ratio` at `[0.50, 0.25, 0.25]`, more concentrated than MLP. Both
> families rank the cards identically (6.2-6.6x), so an independent vector adds
> 0.16 ms of a 245 ms compute optimum (0.01 %) unconstrained, and +0.4 % …
> +2.2 % of prefill under the rank-0 VRAM constraint, against a 3.18 % floor.
> The one bootable step in the proposed direction (`GDN 10,3,3` at MLP
> `10,1,1`) is worth +2.4 %; the next step does not fit on the 5090. Dynamic
> clones are rejected additionally on bytes, swap latency, the lending
> contract, GDN state locality, graph capture, and the absence of a decode-side
> sensor. New reason required to retry: a model where the attention family's
> per-rank speed ratio diverges materially from the MLP family's (e.g. an MLA
> or very-wide-GQA checkpoint, or a card pair where one side lacks fp8 for
> GEMM but matches on bandwidth).

### 8.2 First step, with the expected gain

**Boot arm 2's recipe with a capacity-matched `--rank-kv-ratio`** (§7). One
boot, one flag, no code.

* Expected: `max_total_num_tokens` 69,784 -> ~430,000 (**6.18x**, three orders
  above any noise floor), prefill unchanged at ~1546 tok/s, ms/Verify to be
  measured.
* Effort class: **XS** — one arm on an existing harness
  (`scripts/gpu_battery/s15_phasen_optima.sh` already boots exactly this
  recipe; add `--rank-kv-ratio 2,11,10` or `capacity`).
* Risk: the decode side of an extreme token vector is unmeasured. Measure
  ms/Verify at bs=1 and bs=8 in the same boot; arm 4 is the prior and it points
  the right way.
* Second-order: if it holds, `--rank-kv-ratio capacity` should arguably become
  the default whenever a family vector is pinned, and the planner's
  decode-knee guard should be re-fitted without the phantom context penalty.

### 8.3 If the phase ladder (#274/#287) is picked up again

Use the corrected #296 economics (prefill +14.2 %, ms/Verify +11.1 %, context
**unchanged**), and treat the base `--rank-tp-ratio` as the attention/GDN axis
(§3.3) if anyone wants to sweep it — no flag work needed to measure, and the
measurement will land under the floor.

---

## 9. Why no card window was taken

The briefing allowed up to 20 minutes if the analysis produced a concretely
measurable candidate. It did not, for #299 proper:

* The predicted effect of the only bootable point on the axis (`GDN 10,3,3`) is
  **+2.4 %** against a measured s=8 prefill noise floor of **3.18 %**. Booting
  it would produce a number that cannot carry a verdict either way — precisely
  the case the measurement rules say not to report.
* The one candidate far above the floor (§7, 6.18x) belongs to #296/#274, not
  to #299, and should get its own window with a proper decode arm rather than
  being smuggled into this one.
* The cards were held for the duration of this investigation:
  `/spinning/gpu-arb/holder` -> `session=agent-lane-r8 cards=0,1,2
  purpose=274-r8-lane-spec-concurrent since=2026-07-31T05:06:49Z`, with a live
  TP=3 server on port 30081 (29.3 GiB in use on the 5090). Queueing behind it
  for a below-floor measurement was not justified.

## 10. Reproducing this

```
# geometry check (CPU only)
cd /spinning/wt-299
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=/spinning/wt-299/python \
  /spinning/shvllm/.venv/bin/python -c "
from sglang.srt.distributed.utils import set_tp_partition_ratios, partition_units
set_tp_partition_ratios([28107,16280,16280])
print(partition_units(136,[28107,16280,16280]))   # MLP  -> [63,37,36]
print(partition_units(16, [28107,16280,16280]))   # GDN  -> [8,4,4]
print(partition_units(4,  [28107,16280,16280]))   # kv   -> [2,1,1]
"
```

Measurement inputs: `/spinning/gpu-battery-results/2026-07-30_phasen_optima/`
(`tabellen.md`, `STATUS.md`, `befunde/fp8_objective_audit.md`,
`s15_phasen_optima/{wait/,proofs/}`) and
`/spinning/gpu-battery-results/2026-07-30_hebel_verif/tabellen.md` for the
noise floors. Prior findings: `docs/dev/INTEGRATION_R3_VALIDATION.md`
sections `#252` (:4724), `#264` (:4768), `#199` (:4305 profile table),
`#296` (:11446), and `docs/DESIGN_201_hierarchische_parallelitaet.md:1181-1213`
(the phase-dual entry that already excluded GDN).
