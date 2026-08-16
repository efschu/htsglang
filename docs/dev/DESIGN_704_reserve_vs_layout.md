# DESIGN 704 — the reserve-vs-layout model

Owner: Slot-2. Desk-only. Every number below is read from an instrumented boot;
nothing is fitted except where a fit is explicitly labelled and its residual
published.

**Why this outranks the levers.** The per-rank reserve tracks CUDA-graph
capture, so a rung change forces recapture and therefore changes the reserve. At
3.5-6.7 GiB per rank it is larger than most rung-to-rung pool deltas, and every
`measured=False` rung in the ladder currently carries it unquantified.

---

## 0 — Verdict

**The model is UNDETERMINED on existing data, but not for the reason expected,
and two things are settled:**

1. **Single-factor layer models are FALSIFIED** — not merely unresolved. The
   reserve is *not monotone* in layers.
2. **layers / attention / GDN cannot be told apart on any multiple-of-4 cut**,
   as an exact algebraic identity rather than a sampling limitation.

The recommended next step is **not a fit**. It is the fourth instrument.

## 1 — Correction to my own gate-close reasoning

At the gate I said only the BINDER's reserve is determinate. That is true of the
*pinned* `available_bytes` (which I derived as `pool x cell`), and false of the
**emitted** one. The boot publishes each rank's own capacity — 594,766 /
463,079 / 436,446 tokens, all different — and the world pool is the min taken
afterwards. So `reserve = rest - available_bytes` is determinate on **every**
rank. Three points, not one.

| rank | layers | attn | gdn | budget MiB | **reserve MiB** |
|---|---:|---:|---:|---:|---:|
| PP0 | 28 | 7 | 21 | 31,800 | **6,687.8** |
| PP1 | 20 | 5 | 15 | 18,800 | **3,561.2** |
| PP2 | 16 | 4 | 12 | 19,800 | **5,166.3** |

## 2 — Falsified: any single-factor positive scaling in layers

PP1 and PP2 are **both 3080s**. PP1 carries **more** layers (20 vs 16) and has
**less** reserve (3,561 vs 5,166 MiB). Reserve is therefore non-monotone in
layers, and no `a + b*layers` with `b > 0` can produce the observed triple. The
same holds for attention and GDN counts, which are proportional to layers here.

This is a refutation from three points, and it does not depend on any fit.

## 3 — The confound is exact, not statistical

On the incumbent, `attn = layers / 4` on **every** rank (28/7, 20/5, 16/4 — all
exactly 4.0), and `gdn = 3 x attn`. The three regressors are perfectly
collinear, so the three hypotheses produce **identical residuals to the digit**:

| form | a | b | residuals (MiB) | max abs |
|---|---:|---:|---|---:|
| a + b·layers | 1,628.9 | 164.51 | +452.6, −1,357.9, +905.3 | 1,357.9 |
| a + b·attn | 1,628.9 | 658.04 | +452.6, −1,357.9, +905.3 | 1,357.9 |
| a + b·gdn | 1,628.9 | 219.35 | +452.6, −1,357.9, +905.3 | 1,357.9 |
| a + b·budget | 616.3 | 0.19 | −56.5, −678.0, +734.4 | 734.4 |

This is the **same degeneracy** that made [28,20,16] and [32,16,16]
non-discriminating for the pool divisor. It was broken there by an absolute
value (`cell_size`); here no absolute value is available, because the reserve is
not emitted in decomposed form.

Two-factor forms (`a + b·layers + c·budget`) have three parameters for three
points and fit exactly, with zero residual and zero information. They are
**unidentifiable** and must not be published as models.

## 4 — Recommendation: the fourth instrument, not a fit

Every previous blocker in this ticket dissolved the same way — the sizer already
computed the term and simply did not emit it (budget posts, then
`available_bytes`, then the mamba post's components). This is the fourth
instance.

`derived_rank_auto_reserve_mib` returns **4,160 MiB uniformly** when called
directly with this boot's arguments, while the metal reserves are 6,688 / 3,561
/ 5,166. So the held-back amount is **not** that function's output alone, and
fitting a curve through three points would be modelling a black box whose
contents the process already knows.

**Proposal:** emit what the sizer actually holds back, per rank, at the point it
is applied — the same one-line pattern as the previous three. One boot then
settles the functional form outright instead of a multi-boot regression that
cannot separate collinear regressors anyway.

## 5 — If a boot must discriminate instead: use a non-multiple-of-4 cut

Any cut where `layers/attn` is not uniform breaks the collinearity. [33,15,16]
does: its ratios are 4.125 / 3.75 / 4.0. Predictions per hypothesis, so the
boot is a free calibration point whichever way it lands:

| form | PP0 | PP1 | PP2 |
|---|---:|---:|---:|
| a + b·layers | 7,058 | 4,097 | 4,261 |
| a + b·attn | 6,893 | 4,261 | 4,261 |
| a + b·gdn | 7,113 | 4,042 | 4,261 |
| a + b·budget | 6,658 | 4,188 | 4,378 |

PP0 separates the forms by 165-455 MiB and PP1 by up to 219 MiB — both well
above the ~2k-token boot noise expressed in these units. **PP2 is predicted
identical (4,261) by three of the four forms**, which is the rank2 blind spot
again: it holds 16 layers / 4 attn in every candidate cut.

Note that [33,15,16] no longer needs to be booted for the pool divisor — that
was settled on absolute grounds by `cell_size` — so if it is booted at all, this
is now its reason.

## 6 — Discipline retained

Any rung pool computed with a reserve carried from another layout is
`measured=False` and self-labels as extrapolated, per the ratified split. This
note does not license carrying a reserve across layouts; it quantifies how wrong
that would be — **up to ~1.6 GiB per rank between two ranks of the same card
type in a single boot**, which is larger than most rung-to-rung pool deltas and
is exactly why the ladder's extrapolated rungs cannot gate a window.
