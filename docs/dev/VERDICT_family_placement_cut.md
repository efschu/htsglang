# Family-placement cut: SOLVED, and the objective as stated is unreachable

Date: 2026-08-17. Hermetic (`CUDA_VISIBLE_DEVICES=""`), no boots. Every number
below is measured from the serving checkpoint or read from the live process.

**Verdict: "GDN compute onto the 5090, full-attention/KV onto the 3080s" cannot
be expressed as a pipeline cut. The two halves are the same knob pulled in
opposite directions. The incumbent `[31,17,16]/[7,5,4]` is already ON the
optimal frontier, and the only direction the rig's headroom supports is the
opposite of the one requested.**

## Why it is unreachable, in one line of arithmetic

Pipeline stages are CONTIGUOUS layer ranges -- activations flow forward, so a
stage cannot be a scattered subset -- and the checkpoint interleaves full
attention uniformly at `full_attention_interval = 4` (FA at layer indices
3, 7, 11, ... 63, verified from `text_config.layer_types`: 48
`linear_attention` + 16 `full_attention`). Enumerating every contiguous cut of
the 64 layers gives an exact frontier:

    max_layers_on_a_stage = 4 x (its FA count) + 3

| FA on stage 0 | max layers on stage 0 |
|---|---|
| 4 | 19 |
| 5 | 23 |
| 6 | 27 |
| **7** | **31  <- the incumbent** |
| 8 | 35 |
| 9 | 39 |

So shedding one FA layer from the 5090 costs it exactly **four** layers, three
of them linear/GDN. "More GDN on the 5090" and "less KV on the 5090" are not
two goals that can be traded against each other; they are one monotone knob.
**The incumbent sits exactly on the frontier** -- 31 is the most layers a stage
can hold while carrying only 7 FA layers. There is no slack being left unused.

### What `--pp-attn-stage-ratio` really buys, and its bound

Called through `derive_pp_layer_split` both ways on the real checkpoint:

| flags | layer counts | FA per stage |
|---|---|---|
| `--pp-stage-ratio 31,17,16` alone | `[32,16,16]` | `[8,4,4]` |
| `+ --pp-attn-stage-ratio 7,5,4` | `[31,17,16]` | `[7,5,4]` |

The decoupling is REAL and the incumbent is already exploiting it: without the
flag the boundary snaps to a multiple of the period and the 5090 would carry
**8** FA layers. With it, one FA layer and one total layer come off. But the
gain is bounded by construction -- it moves the boundary WITHIN one period, so
it is worth at most one FA layer per boundary and it cannot break the 4:1
frontier.

(I first read `7,5,4` as the neutral coupled value by counting FA in contiguous
ranges. That was wrong; calling the real derivation settled it. The coupled
derivation of the same scores is `[32,16,16]`, not `[31,17,16]`.)

## The funded numbers

Weights **measured** from the safetensors index via
`planner.pp_cut.checkpoint_weight_terms` -- not derived from config formulas,
which that function's own docstring warns are wrong by ~30 MiB per attention
layer on this family. The measured attention layer is **355.1 MiB**, exactly
the value the docstring cites as the real one.

| term | value | provenance |
|---|---|---|
| FA layer weights | 355.1 MiB | measured, checkpoint index |
| GDN layer weights | 476.1 MiB | measured, checkpoint index |
| embedding / lm_head | 2425.0 MiB each | measured |
| replicated (visual+mtp) | 1283.9 MiB | measured |
| KV per FA layer @ 436275 tok | 852.1 MiB | `kv_bytes_per_token_per_attn_layer = 2048` x tokens |
| max-total-tokens | 436275 | live process args |

**GDN layers are 1.34x HEAVIER than FA layers** (476.1 vs 355.1 MiB). So
concentrating GDN also concentrates weight, not only compute -- which is fine
on the big card and is another reason the requested direction fights itself.

## The three cuts, priced

| cut | attn | GDN per stage | PP0 MiB | PP1 MiB | PP2 MiB |
|---|---|---|---|---|---|
| **INCUMBENT** `[31,17,16]` | `[7,5,4]` | `(24,12,12)` | **19877** | **11750** | **10542** |
| requested direction `[27,21,16]` | `[6,6,4]` | `(21,15,12)` | 17242 | 14385 | 10542 |
| more-GDN `[35,15,14]` | `[8,4,4]` | `(27,11,10)` | 22513 | 10066 | 9590 |
| more-GDN `[39,13,12]` | `[9,4,3]` | `(30,9,9)` | 25148 | 9114 | 7907 |

The requested direction moves **2635.6 MiB off the 5090 and entirely onto a
3080**, and it *reduces* GDN on the 5090 from 24 layers to 21. It is worse on
both halves of the stated objective simultaneously.

Against card capacities (5090 32768 MiB, 3080s 20480 MiB), the incumbent leaves
roughly 12.6 GiB unused on the 5090 and 8.5 / 9.7 GiB on the 3080s. The scarce
headroom is on the SMALL cards, so spending it is the wrong move; the direction
that has room is loading the 5090 further.

### The #690 tension, resolved rather than hand-waved

#690 wants the fewest LAYERS on the x4-attached 3080 (its flip-refill is
bandwidth-bound). The requested direction gives a 3080 **21** layers -- the
most of any stage except PP0 -- so it is worse for #690 too. The more-GDN
direction gives the last stage **14** (or 12) layers, the fewest on the rig.
The two constraints do not actually conflict here: both point away from the
requested cut and toward loading the 5090.

## Flag sets

**Recommended: keep the incumbent.** It is on the frontier, it already exploits
the #485 decoupling, and no alternative improves the stated objective.

    --pp-stage-ratio 31,17,16  --pp-attn-stage-ratio 7,5,4
    --max-total-tokens 436275

**The one arm worth a window** -- the direction the headroom supports, which
delivers the *achievable* half of the user's intent (more GDN compute on the
5090) and relieves both small cards:

    --pp-stage-ratio 35,15,14  --pp-attn-stage-ratio 8,4,4
    --max-total-tokens 436275          # co-solved; see unknowns

GDN on the 5090 rises 24 -> 27 layers; PP1 falls 11750 -> 10066 MiB and PP2
10542 -> 9590 MiB. Cost: one more FA layer of KV on the 5090 (+852 MiB), which
is where the headroom is.

`SGLANG_UNEVEN_TOKEN_VECTOR` is **not** part of either set: it is a TP-token
split and the serving config is `--tp-size 1 --pp-size 3`. See below.

## Honest unknowns -- what only a boot can answer

1. **Per-stage FREE VRAM.** Everything above is weights + KV. Activations,
   CUDA-graph pools, the seam reserve and the corridor are boot artifacts
   (residency census), not checkpoint properties. The solver reports deltas and
   **refuses to claim fit**; whether `[35,15,14]` actually boots at 436275
   tokens is a boot question.
2. **Which physical card is PP0.** Everything here assumes PP0 = 5090. That
   must be resolved through the NVML IdentityMap at boot, never by ordinal --
   torch and NVML orders diverge on this rig.
3. **Whether `--max-total-tokens` holds at the new cut.** The pool is set by
   the tightest stage; the more-GDN cut relieves the 3080s, so 436275 should be
   reachable or exceeded, but the token vector follows the cut and must be
   co-solved rather than carried over (the boot-2 lesson).
4. **Host-RAM per-stage images.** PP0 grows from 31 to 35 layers, so its host
   image grows ~1.8 GiB. Under the per-stage accounting the largest stage is
   already ~2x the smallest; this widens that. Not priced here.

## What PPCutInputs does NOT fund, stated plainly

`PPCutInputs` is genuinely family-aware -- per-layer family tags, per-family
weight bytes and FLOPs, `attn_core_flops_per_token_pair`,
`kv_bytes_per_token_per_attn_layer`. What it does **not** carry is any term for
*where a family should live*: `solve_pp_cut` minimizes the lockstep makespan
over contiguous cuts, with memory feasibility as a constraint and headroom as
the tiebreak. There is no objective weight for "KV on the big card" or "GDN
concentration", and adding one would be inventing a cost term. The frontier
above is the honest substitute: it shows the whole reachable set, so the
objective can be read off it rather than encoded into a solver that has no
funded units for it.

## Relation to #705, which refused a different thing

`d937d5f76b` refused a family split, but that is the **TP-decode** family split
-- concentrating a family's shards on one TP RANK to remove 48 of 128
collectives, priced at +0.090 ms on a ~30 ms round. It is not this question.
Two things follow:

* Its headline alternative ("uneven TP alone is worth +0.780 ms/round, already
  shipped, simply not enabled") **cannot be collected on this config at all**:
  the live server runs `--tp-size 1`, so there is no TP sharding to make
  uneven. `--rank-tp-ratio` is inapplicable, not merely unset.
* Its capacity ledger and its "MoE is 77.7% of weights" premise describe an MoE
  model. The serving checkpoint is dense (no `num_experts`), so those numbers
  do not transfer here either.

So #705 neither prices nor refuses the PP family placement. This determination
is the first time that objective has been solved, and the answer is that the
geometry forecloses it.

## Boot arm, filed for F4-r4's window list

**Arm name:** `family-cut-more-gdn`. One A/B against the incumbent. This is the
only arm the analysis supports; the requested direction is dominated on both
halves of its own objective and no arm is requested for it.

```
# control (incumbent, already serving)
--pp-stage-ratio 31,17,16 --pp-attn-stage-ratio 7,5,4 --max-total-tokens 436275

# treatment
--pp-stage-ratio 35,15,14 --pp-attn-stage-ratio 8,4,4
# --max-total-tokens CO-SOLVED at boot, not carried over (boot-2 lesson);
# the token vector follows the cut.
```

Resolve PP0 -> physical card through the NVML IdentityMap before reading any
result; the analysis assumes PP0 is the 5090 and that assumption is not
self-verifying.

**Acceptance, in order. Stop at the first failure.**

1. **GATE 0 -- it boots and generates.** One-generation smoke, content-sane.
   A cut that does not boot has no timing.
2. **GATE A -- pool.** `--max-total-tokens` co-solved must be `>= 436275`.
   The treatment relieves both 3080s by 1.7 and 1.0 GiB on paper, so a pool
   BELOW the incumbent falsifies the residency model and is the finding.
3. **GATE B -- TTFT.** Prefill is the pipeline-bubble regime (#691); TTFT must
   not regress beyond the boot's own A-vs-A floor, measured first.
4. **GATE C -- decode ms/round per rank.** The predicted effect is a
   redistribution, not a speedup: PP0 does more work per stage while PP1/PP2 do
   less. Report per-rank ms/round, not an aggregate -- the #588 lesson is that
   pooling ranks hides the asymmetry that matters.

**What would falsify the analysis:** PP1/PP2 free VRAM failing to rise by
roughly the predicted 1684 / 952 MiB. That would mean per-stage residency is
not dominated by layer weights + KV, and the whole frontier pricing would need
the boot's census rather than the checkpoint index.

**Explicitly not in this arm:** `--rank-tp-ratio` / `SGLANG_UNEVEN_TOKEN_VECTOR`
(the config is `--tp-size 1`, so there is no TP split to make uneven) and any
change to the prefill graph backend (#613's regime gate is a separate arm; a
captured prefill would confound GATE B).
