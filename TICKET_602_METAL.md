# TICKET 602 — METAL ARM for the KV-floor PP cut

**Status: NO ARM TO SCHEDULE on the censused regime.** Not merely deferred —
its premise is absent there. See §3. Desk work on `fix/602-fill-side`.

> **REVISION 3 (after F4-r4's census boots).** Two earlier revisions of this
> ticket quoted a "predicted world MIN" and gated it at ±5 % against a measured
> pool. **That comparison was a category error** and F4-r4 correctly refused to
> boot on it (-23.3 %). The numbers below are restated by KIND, and the
> recommendation itself is withdrawn for his regime.
>
> Revision 1: `31,16,17`, +227095 tokens (+36.3 %) — withdrawn (bench weights).
> Revision 2: `29,19,16`, +17235 tokens (+3.6 %) — withdrawn (see §3).

---

## 1. The two numbers, and why they are not interchangeable

The solver now emits **two** outputs. They differ by exactly one term and by
~29 % on this rig.

| output | funds the worst load transient? | what it means |
|---|---|---|
| `world_predicted_pool` | **no** | the pool the live sizer will produce |
| `world_corridor_safe_floor` | **yes** | the largest pool that stays corridor-safe *while a cutover is in flight* |

**Only `world_predicted_pool` may be compared against a boot's measured
`max_total_num_tokens`.** The live sizer charges no load transient at all,
because it sizes the pool *before any seam has ever run*. In this regime the
worst load state is a **SEAM on every rank** — `SEAM_TP_TO_PP` 2168 MiB on
rank 0, `SEAM_PP_TO_TP` 700 / 932 on ranks 1 / 2, two to three times the
prefill-triggered scalars — so funding it moves the answer by nearly a third.

Neither number is wrong. Conflating them is, and that is what the ±5 % gate in
revisions 1–2 did.

## 2. Gate, restated

**Compare `world_predicted_pool` against the measured pool, ±5 %.**

Validated on F4-r4's census boots (measured 471303 tokens, incumbent cut):

```
world_predicted_pool        468984    -0.5 %   <- inside the gate
world_corridor_safe_floor   333645   -29.2 %   <- the number that was being
                                                  compared, and must not be
```

The corridor-safe floor has **no** gate against a measured pool. If a
corridor-safety claim is wanted it needs its own arm — a boot that actually
runs cutovers under load — not this comparison.

## 3. Why there is no arm on the censused regime

Re-solved on F4-r4's terms, **the incumbent cut is the global optimum**:

```
[28, 20, 16]   468984   <- incumbent, best of all 1953 contiguous cuts
[29, 20, 15]   439256   -6.3 %
[29, 19, 16]   439256   -6.3 %   <- revision 2's recommendation
[30, 17, 17]   346105  -26.2 %   <- revision 1's shape
```

Nothing beats it. Revision 2's `29,19,16` makes the pool **smaller**, so there
is no cut to arm and no gain to measure.

**The +3.6 % was a property of my boot's terms, not a portable result.** The
two regimes differ materially — attention layer 374.24 vs 362.3 MiB, replicated
payload 920.45 vs 1284 MiB, residuals 4874.6/3001.4/3005.2 vs 1678/482/482 —
and the optimum moved with them. A cut solved on one regime must not be armed
on another.

**Before any future arm: re-run the solve on the regime being booted**, using
that regime's own census, and check `world_predicted_pool` against its measured
pool first. If the incumbent is already optimal there too, the correct outcome
is again no arm.

## 4. Calibration inputs (F4-r4's census, the reference for this ticket)

Source: `/spinning/evidence-665-f1/census-602/` (`census_pp*.json`,
`transient_pp*.json`), solver cherry-pick `299f2922e0`.

| term | value | derivation |
|---|---|---|
| attn layer weight | 374.24 MiB | `params_mib.layers_attention / n_attn_layers`, meaned |
| linear layer weight | 476.21 MiB | same, and identical on all three stages |
| replicated (visual) | 920.45 MiB | `params_mib.visual`; census reports no separate MTP line |
| embed / lm_head | 2425.0 MiB each | stage 0 / stage 2 |
| residual overhead | 4874.6 / 3001.4 / 3005.2 | `nvml_used - params - pools` |
| seam held at rest | 2711.7 / 702.4 / 1134.4 | `nvml_free - 1024` |
| worst transient | 2168 / 700 / 932 | `worst_transient_mib`, a SEAM state on every rank |
| KV cell | 2399.6 B/token/attn-layer | `pools_mib / (tokens x n_attn)`, meaned |

## 5. If an arm is ever scheduled

Unchanged from revision 2 and still binding:

* **Hard**: corridor ≥1024 MiB NVML-FREE per card; the 1728 MiB arming floor
  holds and a flip completes each way; no OOM; health 200.
* **Hard**: `world_predicted_pool` within ±5 % of the measured pool (§2).
* **Must not overlap F4-r4's #689 window** — re-cutting moves per-stage arena
  and seam geometry and invalidates the cached per-rank seam records, which are
  exactly what his peer-fit and arena-occupant work measures.
* The queued #681 proof (`0274bed857`) can share a window: admission-path arm,
  moves no layers.
