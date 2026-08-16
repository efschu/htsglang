# TICKET 602 — METAL ARM for the KV-floor PP cut

**Status:** ready to schedule. **One boot.** Desk work complete on branch
`fix/602-fill-side`.

**Do not run concurrently with F4-r4's #689 acceptance measurements** — see §6.

> **REVISION NOTE.** An earlier version of this ticket recommended
> `31,16,17` and predicted `+227095` tokens (`+36.3 %`). **Both were wrong**
> and are retracted. They came from a cost model whose per-layer weights were
> the #485 reference bench's, not this checkpoint's: the real linear layer is
> 476 MiB, so moving layers onto a stage costs far more than the model
> believed. Calibrating the weights from the checkpoint reversed the
> recommendation. The numbers below are from the calibrated model, which now
> reproduces the floor the live boot actually sized.

---

## 1. What is being tested

Term 2 of the #602 fill-side attribution: the world KV token count is
min-reduced across PP stages
(`model_runner_kv_cache_mixin.py:4585-4592`), so every stage above the minimum
strands its surplus. The only lever that converts it is the layer cut.

**The change is one flag.** No code path is enabled or disabled by this boot;
the solver is a desk tool and none of it runs in the serving process.

## 2. The arm

```
--pp-layer-ratio 29,19,16        # currently 28,20,16
```

Everything else identical to the reference boot.

**Stage 2 keeps 16 layers.** One linear layer moves from stage 1 to stage 0;
the attention split is unchanged at `[7,5,4]`. Every cut that puts 17 layers on
stage 2 prices INFEASIBLE once the weights are real.

## 3. Acceptance

### 3a. HARD

1. **Corridor holds.** ≥1024 MiB NVML-FREE per card at rest (free = the NVML
   FREE column, never total−used).
2. **Arming floor holds.** 1728 MiB free per rank; the flip still arms and
   completes at least one cutover each way. Term 1 stays booked.
3. **No OOM, no wedge**, health 200, smoke request answered.
4. **World MIN tokens > 471638** (the measured baseline at `28,20,16`).

### 3b. HARD FLOOR GATE — now a gate, with a stated tolerance

**Predicted world MIN at `29,19,16`: 497245 tokens. Accept within ±5 %,
i.e. 472383 … 522107.**

The tolerance is earned, not asserted. Against the same boot the calibrated
model predicts the INCUMBENT cut's floor as 480010 where the boot sized
471638 — **1.8 % high** — and reproduces per-stage at-rest residency to
−53 / +8 / +39 MiB (0.2 %). ±5 % is that demonstrated error with margin.

**A measured floor below 472383 fails the boot**: it would mean the reclaim did
not materialise. **Above 522107 also fails** — the model would be
under-pricing something and the calibration is not trustworthy for the next
cut either.

### 3c. Expected size of the prize, stated plainly

```
incumbent [28,20,16]   480010 predicted   (471638 measured)
solved    [29,19,16]   497245 predicted
gain                   +17235 tokens   (+3.6 %)
```

**This is a single-digit-percent improvement.** If a ~3.6 % KV gain does not
justify a boot slot against other queued work, that is a legitimate reason to
defer this ticket — the honest number is the point of the exercise.

## 4. Calibrated inputs (all measured, all provenance-stamped)

| term | source | value |
|---|---|---|
| attn layer weight | checkpoint safetensors headers | 362.3 MiB |
| linear layer weight | checkpoint | 476.1 MiB |
| embed_tokens (stage 0) | checkpoint | 2425 MiB |
| lm_head (stage 2) | checkpoint | 2425 MiB |
| replicated: vision tower + MTP head | checkpoint | 879 + 405 MiB |
| draft runner, net | recorder | 2906 / 2250 / 2250 MiB |
| fixed overhead | recorder | 1678 / 482 / 482 MiB |
| transient (observed) | recorder serving marks, 996 s | 742 / 440 / 584 MiB |
| KV per token per attn layer | recorder kv posts | 2326.7 B |
| seam | fixed point, converged | see §5 |

Weight identity against the recorder: **−53 / +8 / +39 MiB**.

### 4a. Boundaries, named rather than approximated

* **The GDN/mamba state pool is not separable.** The recorder folds it into the
  single `kv_pool_sized` post. The three stages fit that post as linear in
  ATTENTION-layer count with a residual constant of −132 MiB and a spread under
  3 MiB, which leaves no room for a term scaling with LINEAR layers. So
  `state_bytes_per_linear_layer` is 0 on the strength of that fit, and the fit
  is pinned in `test_checkpoint_weight_terms_602.py`.
* **The transient is the observed window, not a soak.** 996 s, 34 samples. The
  #485 bench's 1346/1120/982 MiB came from a different checkpoint and is not
  transferable; charging it makes the RUNNING config price infeasible, which
  is how it was found to be wrong for this deployment. A soak-length recorder
  window should replace this and would move the gate.

## 5. Rollback

Revert to `--pp-layer-ratio 28,20,16` and reboot. Nothing persistent is
written — except that the seam records under
`/root/.cache/sglang/kv_budget-*-seam-rank*.json` ARE cut-sensitive: if the
boot is abandoned, let the next boot re-measure them rather than inheriting a
`29,19,16` record into a `28,20,16` tree.

## 6. Scheduling constraint

**Must not overlap F4-r4's #689 acceptance measurements.** This cut changes
each stage's KV arena size and occupant set, and its seam staging demand (the
per-token seam slope is attached to the stage's attention count), which
invalidates the cached per-rank seam records. Those are exactly the quantities
his peer-fit-asymmetry and arena-occupant work measures; interleaving would
leave both sets unattributable.

The queued #681 proof (`0274bed857`) has no such conflict and can share a
window: it is an admission-path arm and moves no layers.
