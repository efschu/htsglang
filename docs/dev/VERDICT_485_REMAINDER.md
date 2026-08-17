# #485 remainder determination — the matrix, priced against the actuator

**2026-08-17, Slot-3. Determination only; nothing built.** Extends
`docs/VERDICT_485_phase_matrix.md` (4c5e594747) rather than restating it —
that verdict established PARTIALLY DELIVERED; this one prices the remainder
against machinery that landed after it (#704a, #723, #705, Cluster-B).

## 1. What matrix output exists today — re-verified on two branches

| | serving head `88a0d787da` | integration base `feat/706` |
|---|---|---|
| `solve_pp_cut` (`planner/pp_cut.py:851`) | present | present |
| `layer_families_from_config` (`:622`) | present | present |
| `attention_counts` (`:673`) | present | present |
| `token_shares_from_vector` (`:691`, the #492 axis) | present | present |
| `stage_kv_capacity` (`:1273`) | **absent** | present |
| `solve_family_placement` / `family_split.py` (#705) | **absent** | **absent** |

**No function emits a `(family × phase)` structure on either branch.** The
family axis is an **input** to a single-phase solve (`layer_families_from_config`
→ `attention_counts` → `solve_pp_cut`); the phase axis is chosen by *which
solver you call*. The O1 finding of the earlier verdict therefore stands, now
re-verified against the head that actually serves.

Note for the merge train: `solve_family_placement` is on neither branch, so
#705's decode-column solver is not merely dead code — it is **not present in
the shipping lineage at all**.

## 2. The remainder, priced — and most of it is SUPERSEDED

### The finding: per-family flips at phase boundaries are economically dead

The promise was "every family its own optimum **per phase**", which only pays
if a family's placement can be *changed at a phase boundary*. #704a priced what
such a change actually costs:

> A rung/layout change is a **whole-arena host→device refill**, and its cost is
> **constant in the distance travelled** — `PhaseFlipStacks.refill` copies the
> whole boot-baked image either way. Measured inputs give **1575 ms**.

The actuator has no notion of "just one family". Against #705's measured
benefit for the family split *beyond* uneven-TP:

| | |
|---|---:|
| per-family flip cost | **1575.3 ms** (whole arena, distance-independent) |
| family-split benefit (#705, beyond uneven-TP) | **0.090 ms/round** (0.30% of a ~30 ms round) |
| rounds to repay ONE flip | **17,503** |
| = continuous decode at that operating point | **8.8 minutes** |

A phase-flip regime changes far more often than 8.8 minutes, so **a per-family
flip cannot repay itself before the next boundary.** For contrast, uneven-TP
alone (0.780 ms/round, already shipped, #709) repays a flip in ~1.0 minute —
which is why *that* one is worth a window and the family split is not.

This is not a new refusal; it is #705's verdict arriving at the same place from
the cost side. #705 refused the family split on its **benefit** (+0.3% of a
round). #704a independently kills it on its **cost** (whole-arena granularity).
Two independent arguments, same conclusion.

### What SURVIVES, and it is the honest rewrite

The matrix concept is not dead — the *incremental per-family flip* is. A
per-family × per-phase solve remains worth having as a **boot-time layout
chooser**: one arena per phase, families placed within it once. That rides for
free inside a flip that is already happening, so it pays the 1575 ms **zero
extra times**.

**Proposed task rewrite for #485:**

* **CLOSE** — "per-family optimum switchable per phase" (runtime actuation).
  Superseded by #704a's whole-refill granularity plus #705's measured benefit;
  numbers above. Do not re-open without a *finer-grained actuator*, which
  `REACH_NO_WEIGHT_MOVER` (`regime_stages.py:100`) says does not exist.
* **KEEP, rescoped** — "solver emits per-family placement **per phase layout**,
  consumed at boot/flip time". This is the delta worth building, and it
  composes the existing solver rather than adding a second one: give
  `solve_pp_cut` a per-family return shape instead of a per-stage layer count.
* **KEEP** — the family enumeration gap (O3): vocab, experts, nonlinear
  kernels, per-quant-lane linears have no per-phase treatment anywhere. But
  note this is now a *boot-time placement* question, not a flip question.
* **SEPARATE PROGRAMME** — O4 (diffusion, SR/video, TTS/ASR, training tenants).
  Shares nothing with the LLM layout machinery but the law.

## 3. Which families have NO per-phase treatment — unchanged

Delivered: attention vs GDN/linear (the layer-family boundary), with the #492
replication+token-shard axis on the attention row and #503's re-solve on the
real replicated-KV geometry.

**No per-phase treatment anywhere:** KV heads as their own family, vocab,
experts, nonlinear kernels, linear layers per quant lane. Absence claim
evidence: `planner/` on both branches contains only `pp_cut*.py` and
`split_probe.py`; no module names or solves any of those families.

## 4. Scope boundary

The solver/matrix side only. Trigger and actuation belong to the #363
determination on the other lane — and the finding above is precisely that the
*actuation* side has no affordable per-family move, which that lane should have.

## 5. Not established

The 1575 ms is arithmetic from measured link bandwidth (#704a); no rung change
has been performed on metal. The 0.090 ms/round is #705's desk price. Both are
desk numbers, and the conclusion is a *ratio* of two desk numbers — but the
ratio is 17,500×, so it does not turn on their precision.
