# NOTE 492 — the attention family has TWO distribution axes

DESK / PREDICTED. No boot was run for this note. It corrects
`NOTE_485_joint_phase_vectors.md`, whose "the attention family is grid-pinned"
passages are marked REFUTED there rather than deleted.

## 1. The mistake, stated plainly

#485 slice 1 built a candidate space of attention/GDN **head vectors**. On
Qwen3.6-27B at tp=3 the attention head grid is 4 kv-head units with a
">= 1 unit per rank" floor, so every vector in that ladder — `[3,2,2]`,
`[3,1,1]`, `[5,1,1]`, `[8,1,1]`, `[16,1,1]` — materializes the same partition
`[2,1,1]`. The slice observed that correctly and then wrote the conclusion
down about the **family**: "the attention family has no lever on this
checkpoint", "grid-pinned".

That does not follow, and the fork's own machinery is the counterexample.
**kv heads are cloneable.** The distribution law in CLAUDE.md now says so
explicitly: a kv-head count that does not divide across the ranks NEVER
hard-pins the family, and any "grid-pinned" claim that has not priced the
replication axis is invalid.

The falsifier is executed rather than argued —
`scripts/dev/485_joint_phase/backtest_joint.py` §5 enumerates the whole #485
candidate space and counts the attention partitions it realizes:

```
INT8: HEAD axis -- 5 candidates on a 4-kv-head grid realize 1 distinct
      attention partition(s) [(0.5, 0.25, 0.25)]
      TOKEN axis -- 4 candidates [[2,1,1],[4,1,1],[8,1,1],[16,1,1]],
      grid-free (the owner rule takes any positive integer per rank)
      falsifier (head-only space cannot move the family): PASS
```

A search restricted to that axis cannot move the family whatever it does.

**And the evidence was already on the page.** `NOTE_475_phase_prefill_prediction.md`
§4, "Residual honesty", says it in as many words:

> The barrier term accounts for 27 of the 41 ms/1k the `#424` INT8 pair moved;
> the remaining 14 ms/1k is not explained here. That arm also changed
> `--rank-kv-ratio` to `2,11,10`, which redistributes DCP token ownership and
> **therefore prefill attention work** — a second skew source the family model
> does not represent at all. `#433`, which changed only the MLP vector, has no
> such residual.

That is a MEASURED 14 ms/1k of prefill attention work moving with the token
vector, on this rig, before #485 was written. Slice 1 read that note, took its
per-barrier max, and then declared the attention family pinned on the strength
of a head-grid search. The axis was not merely unpriced — it had a measured
residual attached to it and no place in the model to put it.

## 2. What the runtime can actually do today

Checked before building, because "the machinery already handles that" is the
sentence that produced the error. There are **two different replication
mechanisms** in `distributed/utils.py` and they are gated differently.

### 2.1 KV-pool replication + token shard — LIVE, and not gated on kv heads

`uneven_dcp_kv_replicated` (`python/sglang/srt/distributed/utils.py:346-354`):

```python
return dcp_size > 1 and get_tp_partition_ratios() is not None
```

It reads the DCP size and whether a base plan is installed. **It never reads
the kv-head count.** So on this 4-kv-head checkpoint at tp=3 — where
`attn_kv_replicated` is False — the KV pool is *already* replicated-heads +
token-sharded on every `--rank-perf-tune phase-*` boot:

* `model_executor/model_runner_kv_cache_mixin.py:2744-2748` — `_pool_kv_head_num`
  returns `get_total_num_kv_heads()` under exactly this predicate, so each
  rank's pool is shaped for the FULL head set;
* `_dcp_token_sharded_pool_rows` (same file, from :2750) sizes it at this
  rank's `ratio_r / S` share of the global context, not at the whole context;
* `layers/dcp/comm.py:166-217` (`cp_all_gather_heads_uneven`) gathers q/k/v to
  the full head set, and `:228-262` (`cp_lse_ag_out_ar_mha_uneven`) LSE-merges
  and slices this rank's head range back out.

**Consequence, and it is the answer to the briefing's second question:**
every rank runs the attention core over ALL heads against its OWN token
shard. The core's per-rank mass therefore follows the **token vector**, which
is continuous — `cp_token_prefix` takes any positive integer per rank, there
is no unit grid and no ">= 1 head" floor. And replicating the COMPUTE role
costs **no extra KV bytes**: the bytes are token-proportional already, which
is precisely what the briefing suspected and what the code confirms.

### 2.2 Projection-weight replication — NOT available here, named as a posten

`attn_kv_replicated` (`distributed/utils.py:1021-1048`) is strictly
`total_num_kv_heads < tp_size`. Its own docstring records that the `<=` flip
was tried and **reverted on measurement** (Qwen3.5-2B, q=8/kv=2, TP=2: the
#105 ragged kernel rejects the straddling q split at the first request).

On Qwen3.6-27B the geometry makes it unreachable independently of that
threshold. Under REPLICATED-KV the q heads split in units of `kv_total`, so
`units = 24/4 = 6`, and the #116 alignment repair
(`_partition_units_kv_aligned`, `distributed/utils.py:659`) requires

```python
if groups < 2 or groups >= n or units % groups != 0:
    return sizes          # raw split, then the #105 guard rejects it
```

with `groups = 4` (kv heads) and `n = 3` ranks: `groups >= n` holds AND
`6 % 4 != 0`. Alignment is impossible, the raw split straddles, the ragged
kernel refuses it.

**Named posten, deliberately NOT built in this slice:** generalizing
projection-weight replication to `tp <= kv_heads` is a runtime rebuild in the
#169 head-gather family (a ragged kernel with per-rank non-uniform GQA
mapping), not a threshold flip. Rough size: the same order as #105/#116
themselves — a new kernel path plus its straddle handling, its own
determinism gates, and a draft-side story. It is not a prerequisite for
anything in §3: §2.1's axis is live today.

## 3. What shipped

`PerfCostModel` gains the family's second term:

* `AttnCorePlan(token_vector, share)` and `_with_core` — the `_with_attn`
  discipline extended, so a solve with no core plan executes the identical
  pre-#492 float operations.
* `token_shard_fractions(v)` — `v[r]/sum(v)`, exact, grid-free.
* `attn_core_crossover_tokens()` — `attn_proj_params_per_layer /
  (2 * q_heads * head_dim)`. Pure geometry, no fitted constant: the depth at
  which the core's per-token FLOPs (`4 * q * d * S`, `QK^T` + `PV`) equal the
  projections' (`2 * P`). **8,533 tokens** on Qwen3.6-27B.
* `per_family_prefill_compute_times(..., core=...)` — redistributes `share`
  of the ATTENTION family's mass by the token vector. Keyed on the family
  NAME, not on the shard: `draft_attn` shares the `"attn"` shard and must NOT
  follow the target's token vector (§4).
* `predict_capacity(..., token_vector=...)` — prices a PINNED vector at
  `cp_token_context_budget`, i.e. what the owner rule actually funds, so the
  axis's capacity cost reaches the fundability gate.
* `_attn_token_candidates`, `_attn_head_axis_is_pinned`,
  `_replication_axis_lines`, `_cand_token_vector`.

The solve REPORTS the axis and does not install it — same rule as #485 §6.
Both actuators exist (`--rank-tp-ratio` for the head half, `--rank-kv-ratio`
for the token half); writing either from a desk prediction is what #485
declined to do and this slice declines it for the same reason.

### 3.1 The bracket, and why it is a bracket

The core share is the attended context depth, which this parse-time model does
not carry. #485 bracketed the LANE for exactly this reason and inventing a
share here would put the fitted constant back. So the solve runs both ends:

* **CORE-FREE** (share 0) — the attention family is all projections. This is
  byte-identically the #485 model.
* **CORE-PACED** (share 1) — the core paces the barrier, so the family follows
  the token vector.

and prints `CORE-INVARIANT` or `CORE-SENSITIVE`. The crossover above says
which side an operating point is on without the model asserting one.

Stated limitation, because it cuts against the result: the CORE-PACED endpoint
**redistributes the family's existing mass and does not grow it**, exactly as
the lane bracket varies only rates. The real core mass grows as `S/8533`, so
at deep context the endpoint is a LOWER bound on the axis, not a centred
estimate.

## 4. The spec cross-charge

The replication axis has a measured cost in the speculative path (#108) and it
is carried, not ignored — as a named refusal rather than a fabricated
time-shaped malus, because the measurement is an ACCEPTANCE rate and the
target's barrier and the draft's are different rounds.

The #108 rule is two-sided and both sides are measured
(`TASK_108_DRAFT_KV_DCP.md`, "TP > num_kv_heads window" Q2):

* `TP > kv_heads` → `dcp`; `replicated` is the DEGRADED layout there
  (accept 1.05, **61 verify rounds for 64 tokens**).
* `TP <= kv_heads` (this checkpoint) → `replicated`; `dcp` is VRAM-neutral and
  costs **10-16 % acceptance**.

So the axis solves the TARGET's token vector and the draft is deliberately not
dragged onto it. Enforced in code, not only in prose: `_ATTN_CORE_FAMILIES`
is `("attn",)`, and `test_the_draft_attention_family_is_not_dragged_onto_the_token_axis`
fails if a refactor keys the core term on the shard instead. Registered:
`planner/rejected.py::draft_kv_dcp_below_kv_threshold`.

## 5. The numbers — and the verdict is not the one #485 would predict

Desk fixture (`backtest_joint.py`, the #475 rate fixture, COMPUTE only —
those budgets fund no pool):

```
INT8   CORE-FREE   4,1,1 + attn/GDN 3,1,1                 +4.8%
       CORE-PACED  4,1,1 + attn/GDN 3,1,1 + KV tok 4,1,1  +5.2%   CORE-SENSITIVE
FP8    CORE-FREE  10,1,1 + attn/GDN 16,1,1               +25.2%
       CORE-PACED 10,1,1 + attn/GDN 16,1,1 + KV tok 16,1,1 +27.4% CORE-SENSITIVE

detuned-token falsifier (aligned vs reversed, same everything else):
INT8  [4,1,1] +5.24%  vs  [1,1,4] +3.05%   PASS (+2.19 points)
FP8  [16,1,1] +27.38% vs [1,1,16] +18.23%  PASS (+9.16 points)
```

On the **real** rig fixture, where the capacity gate is live, the verdict
inverts — and this is the finding that matters:

```
INT8 (loose 0)  CORE-FREE +5.8% ctx ~318938 | CORE-PACED +5.8% ctx ~318938
                CORE-INVARIANT
                capacity price: ALL 4 token vectors REJECTED by the context
                floor (318938); the most concentrated funds only ~14454.
FP8  (loose 0)  same shape; the most concentrated funds only ~4680.
```

The axis only becomes reachable at a very loose context floor, and then it is
small:

```
INT8 loose 95   CORE-PACED 4,1,1 + attn/GDN 3,1,1 + KV tok 4,1,1
                +6.1% for ctx ~19290 against ~318938   (+0.3 points, 16x ctx)
FP8  loose 80   CORE-PACED 10,1,1 + KV tok 4,1,1
                +19.4% for ctx ~76980 against ~329780  (+0.1 points, 4.3x ctx)
FP8  loose 95   CORE-PACED 16,1,1 + KV tok 16,1,1
                +21.2% for ctx ~34056 against ~329780  (+1.9 points, 9.7x ctx)
```

**The corrected verdict.** The attention family is NOT grid-pinned — it has a
continuous second axis that is live in this fork today. What blocks that axis
on this rig is **capacity, not the kv-head grid**: the weighted owner rule
funds `min_r(P_r / v_r)` blocks, so concentrating the token vector onto the
fast card throws the slow cards' pools away. That is a completely different
statement from #485's, it names a different knob
(`--rank-perf-loose-ctx-percent`, not "a checkpoint with more kv heads"), and
it would have been invisible to a search that never priced the axis.

## 6. Status of the #485 predictions

`TICKET_485_int8_joint_arm.md`'s band (+3.3 % … +6.2 %, point +4.8 %) is the
**CORE-FREE endpoint** of the wider bracket and is unchanged. At the ticket's
own operating point (ctx 131072, `--rank-kv-ratio` pinned to the solve's
matched vector, loose 0) the token axis is refused on capacity, so the ticket
does not need a second arm to be decidable. A REPLICATION arm is only worth
booting if an operator is willing to trade an order of magnitude of context,
which on this rig is not a trade anybody has asked for.

## 7. Open, named

1. Projection-weight replication at `tp <= kv_heads` (§2.2) — a #169-family
   runtime rebuild. Not built, not silently assumed.
2. The core mass at depth is a LOWER bound in the CORE-PACED endpoint (§3.1).
   Growing it correctly needs the attended-depth term, i.e. the same
   measurement the lane bracket already names.
3. The capacity/compute trade on this axis is exactly the shape #363's
   `regime_switch` rungs exist to decide (short-context prefill bursts could
   afford a concentrated token vector that a long-context session cannot).
   That belongs there, not here.
4. Whether a rig with a WIDER context budget per slow card reaches the axis at
   loose 0 is a one-line re-run of §5 with different budgets — the model is
   already general over it, and no boot is needed to answer it.
5. NOTE_475's unexplained **14 ms/1k** on the `#424` INT8 pair (§1) is the one
   MEASURED number this axis could claim. This slice does not claim it: the
   CORE-PACED endpoint redistributes mass rather than growing it, so it cannot
   reproduce a residual of that size, and asserting the match would be exactly
   the fitted constant the bracket exists to avoid. Reproducing it needs the
   attended-depth term (item 2). It is recorded here as the axis's outstanding
   measured anchor, not as a result.
