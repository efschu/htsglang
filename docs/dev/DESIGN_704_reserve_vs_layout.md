# DESIGN 704 — the holdback: CONFIRMED MODEL

Owner: Slot-2. **Status: SOLVED on metal.** All fitted candidates in earlier
revisions of this document are retired, as is the [33,15,16] discriminator
fallback. The form is exact, deterministic, and needs no regression.

---

## 0 — The model

    allowed_tokens = id_space + (free_at_measure - arming_floor - margin) / cell
    adjusted_bytes = allowed_tokens * cell
    holdback_frac  = 1 - allowed_tokens / (profiled_bytes / cell)

`cell = attention_layers_on_rank * 2048 B` (fp8_e4m3, config-derived).

Source: `phase_flip_seam_reserve.floor_allowed_tokens` — *"largest id space
whose RESTING FREE COLUMN still holds the arming floor"* — reached via
`seam_adjusted_budget_bytes` from `_seam_adjusted_budget`, the single funnel
every sizing path crosses.

**Verified against the instrumented boot on all three ranks:**

| rank | profiled/cell (raw T) | allowed | model holdback | reported | slack above floor |
|---|---:|---:|---:|---:|---:|
| PP0 | 1,083,904 | 594,614 | 45.141 % | **45.143 %** | 2,177.7 MiB |
| PP1 | 827,699 | 462,918 | 44.072 % | **44.074 %** | 269.4 MiB |
| PP2 | 1,097,728 | 436,278 | 60.256 % | **60.258 %** | 7.4 MiB |

Agreement is 0.002 pp — the residual is MiB rounding in the logged budget.
`id_space = 435,334`, identical on all three ranks.

## 1 — The three questions, answered

**What computes 45.1 / 44.1 / 60.3 %?** Not a memory reserve in the intuitive
sense. The pool is CAPPED so that the rank's resting free column still holds its
arming floor — an equality, solved, not a subtrahend approximated. The percentage
is whatever that cap costs relative to the raw byte capacity, so it is a
consequence, not a parameter.

**Why does the PP pass pay and the TP pass not?** The charge exists so a flip
can ARM. The PP-phase pool is the layout a flip departs from, so it must rest
above the floor; the TP-stack pass has no flip to arm from that layout and
spends its whole profiled budget — confirmed by `holdback = 0.000 %` on all
three ranks in the second sizing at 22:13:27.

**Why is the BINDER the largest fraction?** Because both extremes meet on PP2:

* it has the **smallest cell** (4 attention layers -> 8192 B), so its raw byte
  budget is the LARGEST token capacity of the three (1,097,728);
* its resting free column sits **7.4 MiB above its arming floor** — essentially
  exactly on it — so `allowed` is pinned at barely more than `id_space`.

Largest raw capacity divided by the tightest floor slack gives the largest
holdback. It binds for the same reason it holds back most, which is why the two
facts always co-occurred and looked like a coincidence.

## 2 — Why every earlier attempt failed

The quantity was never a graph-capture reserve, which is what I kept modelling.
Three external re-derivations missed by +20 %, −3.8 % and −12 %; a fit over
three points was falsified (non-monotone in layers) and confounded (`attn =
layers/4` exactly, so layers/attn/gdn gave identical residuals). All of that was
curve-fitting a term the process solves in closed form. The fourth instrument
made it readable in one boot.

`derived_rank_auto_reserve_mib`'s uniform 4,160 MiB was never this number and
never claimed to be — it is a different reserve entirely.

## 3 — Hand-off: exact rung pools for the ladder

Slot-3's rung pools can go from `extrapolated` to **exact**, without booting
each rung, because every input is either config-derived or shifts computably:

* `cell` — config (`kv_mib_per_token_per_attn_layer_from_config` x attention
  layers on that rank for the candidate cut);
* `id_space`, `free_at_measure`, `margin` — the seam record of the CURRENT boot;
* `arming_floor` — the #676 solver, per layout, already required by
  `LadderInputs`;
* the layout shift — `free_at_measure` moves by the weight and mamba deltas
  between the current cut and the candidate, both already established:

      free_at_measure(cut) = free_at_measure(booted)
                             - delta_weights(cut)      # per-family census
                             - delta_mamba(cut)        # 51.20 MiB/GDN-layer

  and `delta_weights` uses the per-family figures (374.2 MiB full-attention,
  476.2 linear).

So a rung's pool is computable from one booted layout plus config. **What
remains extrapolated is only `free_at_measure`'s shift**, which is arithmetic
over two measured constants rather than an unknown reserve — a far smaller
claim than "carry another layout's capture behaviour", and it should be labelled
as such rather than as `measured`.

## 4 — Retired

* the fitted candidates (`a + b*layers`, `attn`, `gdn`, `budget`) — superseded;
* the [33,15,16] prediction table and the request that it be booted to
  discriminate. The form is settled, so that boot has no calibration value. The
  rank2 blind-spot note is preserved below because it still governs TIMING
  calibration, which this does not touch.

**Preserved — rank2 blind spot (timing only):** every candidate cut keeps rank2
at 16 layers, so a boot pair that does not move it cannot identify its timing
term at any sample size (`calibration_coverage` returns `inf`). That is
unaffected by this result, which concerns the pool.
