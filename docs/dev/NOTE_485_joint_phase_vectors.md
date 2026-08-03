# NOTE 485 — the prefill phase is cut per FAMILY (slice 1: the prefill column)

DESK / PREDICTED. No boot was run for this note. The deciding arm is
`TICKET_485_int8_joint_arm.md`.

The law this implements (CLAUDE.md, *PER-FAMILY x PER-PHASE OPTIMA*): every
weight family has its own optimum per phase, so a single-family arm is a
diagnostic and never a phase verdict. Slice 1 delivers the **prefill column**
of that matrix — rows = families, columns = phases. The decode column is
untouched here.

---

## 1. Why the per-barrier max makes the problem separable

#475 replaced the prefill compute term with

```
t_lockstep = sum_family ( max_rank t[family][rank] )
```

because a prefill step is two all-reduces per layer, not one barrier at the
end. That form has a property the old `max_rank sum_family` did not: it is
**separable over families**. Each family's contribution to the round is its
own barrier's maximum, and no other family's vector appears in it. So the
minimization decomposes:

```
argmin over (v_mlp, v_attn) of  sum_fam max_r t_fam,r
  =  ( argmin over v_mlp of max_r t_mlp,r ,  argmin over v_attn of max_r t_attn,r )
```

subject only to the constraints that genuinely couple them — the shared VRAM
budget, the unit grids, and the context floor.

Two consequences, and they are the whole slice:

* **The optimum of each family is its own lane's rate-proportional split.**
  `max_r (p_fam * frac_r / rate_fam,r)` is minimized at `frac_r ∝ rate_fam,r`.
  Nothing about the other families enters.
* **Compensating one family's imbalance with another family's vector is
  exactly what manufactures barrier skew.** A single-family solve has no
  choice but to compensate: it is asked to minimize the round with only the
  MLP vector, so it loads the MLP onto the rank that is fastest overall,
  which is not the rank pacing the attention barrier. #475 measured the
  price of that at 27.9 ms per 1000 prompt tokens.

At perfect per-family balance every rank is (tied) slowest in every family,
so the Jensen gap is zero: **a fully aligned pair has skew 0 and minimal
lockstep time simultaneously.** That is the shape to expect, and §4 records
where this rig cannot reach it.

## 2. Why #299's verdict does not transfer

ANALYSE_299 priced a free-standing attention/GDN vector at **0.16 ms of a
245 ms compute optimum (0.01 %)** unconstrained, and +0.4 % … +2.2 % under the
VRAM constraint against a 3.18 % floor. That verdict was computed under the
pre-#475 cost model — a model in which the lockstep max is taken **once at the
end of the step**, and in which aligning two families' pacers is therefore
worth exactly zero **by construction**. It measured the only thing its
objective could see: the change in `max_rank sum_family`, which a
weight-conserving re-split of one family barely moves.

What #299 got right and this note keeps: the GDN state pool moves with the
GDN units (~4.7 MiB per request per unit), so a joint cut that concentrates
GDN also concentrates that pool and spends KV capacity. That is modelled here
(`mamba_pool_bytes_for`), and it is why several joint pairs are refused by the
fundability gate rather than by the objective.

`planner/rejected.py::mlp_concentration_611` is unaffected: it is a verdict
about ONE vector serving BOTH phases, which is orthogonal to how the prefill
phase is cut internally.

## 3. What a candidate pair is

`(mlp_vector, attn_vector)`.

The **MLP half** is unchanged (`_mlp_candidates`).

The **attention/GDN half** (`_attn_candidates`) is a score^alpha ladder over
the family's own lane rates, at several resolutions including the GDN grid's
own. It has no counterpart to `_mlp_candidates`' *balance ladder* — that
ladder exists to absorb the other families' fixed work into the MLP vector,
i.e. to compensate, which is the operation the decomposition removes.

Constraints hit, in the order they bite on Qwen3.6-27B (tp=3):

| constraint | value here | effect |
|---|---|---|
| attention grid (#62/#116) | `attn_units = 4` kv heads | every rank keeps >= 1 unit, so the ONLY representable split is `[2,1,1]` — the base. **The attention family has no lever on this checkpoint.** |
| GDN grid | `gdn_units = 16` k heads | the resolving grid; the whole ladder lives here |
| MLP grid | 136 quant-group units | unchanged |
| GDN state coupling (#299) | ~4.7 MiB/req/unit | pool follows the units; priced into `predict_capacity` |
| per-layer family table (#371) | `layer_census` | attention mass is `attn_layers`, not `n_layers`; the vector partitions what the census says exists |
| coupled KV (#435) | matched vector per candidate | each PAIR gets its own matched token vector; the installed layout keeps its own |

So on this checkpoint "the attention/GDN vector" is in practice **the GDN
vector**, and the note says so rather than implying a lever the kv-head grid
cannot express.

Rates come from `GemmScores.resolve(GEMM_FAMILY_ATTN_GDN)` — the #324
per-(rank, family) axis — never from the MLP lane and never by hand.

## 4. The numbers (`scripts/dev/485_joint_phase/backtest_joint.py`)

### 4.1 Regression — the measured points do not move

```
arm                                     pre-#475   shipped  measured  skew ms/1k
FP8  base -> 10,1,1  (#424, NCCL)         +18.3%    +18.3%    +15.2%        0.0
FP8  base -> 10,1,1  (#435, BAR1)         +18.3%    +18.3%    +18.0%        0.0
INT8 base -> 10,1,1  (#424)                +5.8%     +1.9%     -1.2%       27.0
INT8 base ->  8,1,1  (#433, solved)        +6.2%     +2.2%     +1.8%       27.0

rms error (points)                                     2.2
```

Identical to NOTE_475 §4, to the printed digit. It has to be: those four arms
carry no attention vector, and with `attn_vector=None` the arithmetic executes
the same float operations (asserted, not assumed —
`test_joint_phase_vectors_485.py::test_passing_no_attention_vector_is_the_pre_485_arithmetic`).

### 4.2 The pair space, at the boots' own rates

```
INT8   base            skew  0.58 us/tok
       MLP-only best   [4,1,1]            +3.84%   skew 5.69
       JOINT best      [4,1,1] + [3,1,1]  +4.84%   skew 5.55   GDN units [10,3,3]
       joint delta                        +1.01 points

FP8    base            skew  0.00 us/tok
       MLP-only best   [10,1,1]           +18.27%  skew 0.00
       JOINT best      [10,1,1] + [16,1,1] +25.18% skew 9.53   GDN units [14,1,1]
       joint delta                        +6.91 points
```

Mechanics, per family, INT8 base -> joint (us per token):

```
family   base per-rank            joint per-rank           max
attn     4.10 /  7.45 /  7.61     unchanged                 7.61  (grid-pinned)
gdn     13.60 / 24.71 / 25.22    17.01 / 18.53 / 18.91     25.22 -> 18.91
mlp     38.78 / 82.72 / 82.15    56.01 / 51.42 / 50.20     82.72 -> 56.01
```

Note what the joint gain is NOT: it is not skew removal. Skew barely moves
(5.69 -> 5.55) because the attention family is pinned to `[2,1,1]` and keeps
pacing on rank 2. The gain is the GDN barrier's own maximum falling 25 %,
which the MLP vector cannot touch at any value. On FP8 the optimizer goes the
other way and ACCEPTS skew (0 -> 9.53) to buy a much larger GDN rebalance —
the objective is the lockstep time, not the skew, and the two only coincide
when every grid is fine enough to balance every family.

### 4.3 The lane bracket — the honest size of the lever

The attention/GDN barrier is part GEMM (qkv/o, the GDN in/out projections)
and part bandwidth (flash, chunked scan). The **mass split between the two
depends on the context length**, which this parse-time model does not carry.
Estimating it by hand would put a fitted constant inside the one term #475 got
right by having none, so it is **bracketed** instead: the same solve is run
with the attention/GDN rows of the rate table set to the pure GEMM lane and to
the measured #231 GEMV lane, rescaled so the family's total time at the base
plan is unchanged (only the inter-rank RATIO varies, never the mass).

A third, physical point sits inside the bracket: the GDN family is
BF16-resident in both checkpoints (2.0 B/param in the family table), so its
real lane is the dense bf16 probe (3.7:1) and not the checkpoint-wide
quantized one #324 assigns it (3.7:1 int8, 9.8:1 fp8 Marlin). That is a
genuine gap in `checkpoint_compute_format_families`, surfaced by this slice
and not fixed by it.

```
INT8
  GEMM lane (shipped)   MLP-only [4,1,1]   +3.84%   JOINT +[3,1,1]  +4.84%   +1.01 pts
  bf16-resident GDN     MLP-only [4,1,1]   +3.43%   JOINT +[3,1,1]  +6.17%   +2.74 pts
  bandwidth lane        MLP-only [4,1,1]   +3.88%   JOINT +[3,1,1]  +3.33%   -0.55 pts

FP8
  GEMM lane (shipped)   MLP-only [10,1,1] +18.27%   JOINT +[16,1,1] +25.18%  +6.91 pts
  bf16-resident GDN     MLP-only [10,1,1] +18.41%   JOINT + [3,1,1] +20.87%  +2.46 pts
  bandwidth lane        MLP-only [10,1,1] +19.14%   JOINT + [5,2,2] +18.79%  -0.35 pts
```

The solve prints `LANE-INVARIANT` or `LANE-SENSITIVE` per run. On this rig it
is LANE-SENSITIVE for INT8 at the phase-prefill operating point and
LANE-INVARIANT for FP8 (where the joint pairs are refused on capacity, so
there is nothing left to disagree about). A LANE-SENSITIVE verdict is not a
failure of the model — it is the model naming the measurement it needs.

### 4.4 Falsifiers, executed

* **Detuned attention (can-fail, executed and reverted).** The optimal MLP
  vector paired with the aligned attention vector REVERSED — same grid, same
  unit count, mass pushed onto the ranks the lane rates call slowest.
  INT8 `[3,1,1]` +4.84 % vs `[1,1,3]` −2.81 % (7.66 points apart);
  FP8 `[16,1,1]` +25.18 % vs `[1,1,16]` −3.09 % (28.27 points). An objective
  that ignored the second half of the pair would return the identical number.
* **Mechanism disabled (can-fail, executed and reverted).** With
  `_shard_fractions` forced back to `attn_plan = self.base_plan`, 11 of the
  new suite's cases fail (31 of 50 subtests survive) and the backtest script
  refuses outright.
* **Equivalence.** An attention vector `A` must price exactly as rebuilding
  the model on base plan `A`, because `--rank-tp-ratio` is its only actuator.
  Asserted for `attn`, `gdn` and `gdn_base` across four vectors.
* **Generality.** A symmetric RATE profile proposes no attention candidate at
  all. But equal cards behind an UNEQUAL base plan DO have a lever
  (+1.5 % here), which is the correct generalization: the lever is a property
  of *base plan vs lane rates*, not of card heterogeneity.

## 5. What shipped

* `PerfCostModel._shard_fractions(shard, mlp_vector, attn_vector=None)` — the
  attention, GDN and vision families follow the given vector. `None` is the
  pre-#485 branch, byte-identical.
* `attn_vector` threaded through `per_rank_weight_bytes`, `predict_capacity`,
  `residual_free_mib`, `streamed_bytes`, `gdn_unit_partition`,
  `mamba_pool_bytes_for`, the whole prefill cost path, and the decode cost
  REPORT (the decode solve never passes one — slice 2).
* `_attn_candidates`, `_attn_partition_key`, `_cand_vectors`, `_cand_label`,
  `_attn_lane_bracket`, `_with_attn`, `_shard_fractions_of`.
* The phase arms solve over PAIRS and report: the joint per-family line, a
  per-candidate `pacers` field, a separate `candidate PAIR (#485)` block, the
  lane-bracket verdict, and a `JOINT PREFILL LAYOUT` line with a
  copy-pasteable launch command.
* `test/registered/unit/planner/test_joint_phase_vectors_485.py` — 14 tests /
  50 subtests. Full planner suite 2184 passed, 1 skipped, 299 subtests.
* `scripts/dev/485_joint_phase/backtest_joint.py`.

## 6. What deliberately did NOT ship

**The install.** The solve names the pair; it does not write it. The runtime
actuator for an attention/GDN vector is the base plan (`--rank-tp-ratio`):
only `"mlp"` is a named family plan in
`distributed/utils._TP_PARTITION_FAMILIES`, so re-pointing the base plan is
what moves attention heads, GDN k-heads, the SSM pool and the vision tower
together. Writing it from a DESK prediction would change the base split of
every `phase-prefill` boot on the strength of a number no card has seen. The
launch line is printed instead, and the GPU arm decides whether slice 2
installs by default.

One consequence of that choice, stated because it is a real seam: an explicit
`--rank-tp-ratio` takes the pin path, so the #435 coupled-KV seed is not
reachable. The launch line therefore pins the pair's own matched KV vector
explicitly. That is the #435 rule honoured — the boot runs the vector the gate
accepted — not bypassed.

**The flash/scan core mass.** Bracketed, not estimated (§4.3).

**The decode column, and the expert families.** Slice 2 and later;
`expert_compute_placement.py` (#439) is untouched.

## 7. Open, named

1. `checkpoint_compute_format_families` does not report the BF16-resident GDN
   family as diverging, so #324 hands it the checkpoint-wide lane. The
   bracket contains the right answer but does not know it is the right
   answer. Fixing it would narrow the bracket to a point on both checkpoints.
2. The attention family is grid-pinned at `attn_units = 4` on this
   checkpoint. A checkpoint with more kv heads (or DeepSeek V4's `o_groups`)
   would resolve it, and this machinery already handles that — untested on
   hardware.
3. Whether the joint layout is worth its restart at all is a decision-layer
   question that belongs with #363's `regime_switch` rungs, not here.
