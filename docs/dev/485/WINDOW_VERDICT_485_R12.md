# #485 METAL TICKET §7a — WINDOW VERDICT, shift `485-metal-r12`

Window 2026-08-14 09:10Z–10:24Z. Tree: the MERGE-R12 line
(`b275fc6c42` → `7ed4437cd5`); boots ran from `/spinning/wt-merge-r12` (M0, M1)
and `/spinning/wt-485-fixes` (W1). Evidence `/spinning/evidence-631/m485/`.
All boots on port **30041** so the router on 30099 could never reach one.

**Headline.** M0 → M1 → P2 → W1 all ran. Q2 is **decided: S3**, and it is
decided on the one column the analysis nominated, over 594 flips. P2 is the
result that changes the ticket: `--pp-solve-cut` **could not execute at all** —
three defects, two now fixed, the third a missing measurement pass — and once
fed borrowed measured rates it admits **a different cut**, `37,14,13`, not the
`40,12,12` two windows were spent on. W1 certified that cut CLEAN at a
**+2560 MiB** binding margin, and by the protocol's own rule remains
`NOT CERTIFIED (C2)`: one window cannot bound a variance.

---

## 1. THE FOUR QUESTIONS OF §7a

### Q1 — the virtual reserve, and the correction the ticket needs

`PYTORCH_CUDA_ALLOC_CONF` is **UNSET** in the ship environment
(`m485/ship_env_live.txt`, captured from `/proc/1047568/environ` before the ship
was stopped), so `expandable_segments` is off. **That removes the explanation
the analysis offered.** EXCURSION_ANALYSIS §4 attributed the overshoot to
"expandable-segments virtual reservation"; it cannot be that here.

Confirmed, and then extended in the direction that matters:

| rank | card | `nvml_total` | `reserved` | `allocated` | `allocated − total` |
|---|---|---|---|---|---|
| 0 | RTX 5090 | 32088.5 | 33698.0 | 33298.0 | **+1209.5** |
| 0 (M0, post-soak) | RTX 5090 | 32088.5 | 35452.0 | 35052.0 | **+2963.5** |
| 1 (M0, post-soak) | RTX 3080 | 20054.9 | 23646.0 | 23250.1 | **+3195.2** |
| 2 (M0, post-soak) | RTX 3080 | 20054.9 | 22032.0 | 21639.3 | **+1584.4** |

**`allocated` exceeds the card's physical total on all three ranks.** So the
analysis's premise for choosing that column — "live torch bytes, physically
backed **by definition**" — is **false on this runtime**, and the correction is
not cosmetic: it is the same mechanism as the Q2 answer.

`phase_flip_spill.py:447-449` names it. The arena reserves its VIRTUAL range
once with `cuMemAddressReserve` at boot and maps/unmaps physical handles
underneath with `cuMemMap`/`cuMemUnmap`/`cuMemCreate`. Both layouts' tensors
live in that reservation, so torch counts them all as `allocated` while only
the active layout's pages are committed. **Q1 and Q2 are one finding, not two.**

The discriminator still works — see Q2 — because it turns on whether
`allocated` MOVES, not on what it equals. But it is worth stating plainly that
`allocated` is not a physical quantity on this rank either.

### Q2 — S3 vs {S1, S2}: **S3 WINS, unambiguously**

Across every `weights_refill` stage, in both directions, on all three ranks:

| window | rank/dir | n | `free` delta (modal) | `allocated` delta | verdict |
|---|---|---|---|---|---|
| M1 | 0 / `tp_to_pp` | 6 | **−4150** ×5 | **0 .. 0** | S3 |
| M1 | 1 / `pp_to_tp` | 6 | −2412 ×5 | 0 .. 0 | S3 |
| W1 | 0 / `tp_to_pp` | 99 | **−3062** ×97 | **0 .. 0** | S3 |
| W1 | 1 / `pp_to_tp` | 99 | −1644 ×97 | 0 .. 0 | S3 |
| W1 | 2 / `tp_to_pp` | 99 | −338 ×96 | 0 .. 0 | S3 |

**`max |Δallocated|` is 0 MiB — not small, zero — across 594 parsed seam-census
flips on two different cuts.** S1 (allocator segment growth during the checksum
leg) and S2 (a concurrent torch actor) both require `allocated` to RISE with the
drop. It does not move by a single MiB. Both are excluded; **S3, the VMM commit
in `arena_carrier.set_active_prefix`, is the mechanism**, and Q1 independently
shows the arena is exactly such a VMM reservation.

Measured two ways that agree: an extractor written here
(`m485/q2_alloc.py`) and the branch's own `excursion_485.py census`, which
reports the identical flip counts and the identical `weights_refill` step
values (M1: −4150 ×24, −4278 ×1).

*Instrument note.* The first version of that extractor matched **zero** stages,
because its regex demanded `free=N(step±M)` while the runtime prints
`free=N step±M` — a space. It reported "n=0, verdict −" and looked like a clean
run of nothing. It was fixed and re-run before any conclusion was drawn; this
is the R11 §1 shape again and is recorded rather than quietly corrected.

### Q3 — the left term of the identity

At-rest TP free on the binding rank (rank 0, the 5090):

| window | cut | at-rest free | ≥ 8079 MiB? |
|---|---|---|---|
| M0 | ship `14,10,8`, pool 620000 | **3953.7 MiB** | **no** |
| M1 | `40,12,12`, pool 280000 | **2691.7 MiB** | **no** |
| W1 | `37,14,13`, pool 280000 | **4435.7 MiB** | **no** |

Stated even though the answer is plainly no, as §7a requires. The identity's
8079 MiB requirement is derived from the POOLED worst transient (7055), and no
configuration measured here comes near it at rest. **But for W1 that comparison
is the wrong one** — see §3 — because 7055 belongs to a different cut, and W1's
own worst transient is 3186.

### Q4 — the 5536/7055 outlier does **not** recur

Pooled with `excursion_485.py census` over s50 + s51 + M1:

```
   flips matched: 221   (196 prior + 25 from M1)
   modal transient  5800 MiB     worst 7055 MiB     2nd worst 6008 MiB
   worst - 2nd worst  1047 MiB
```

**The pool grew 196 → 221 and the worst did not move.** It is still s50's single
event at 11:44:06. M1 contributed 25 flips whose worst is **4378 MiB**, 2677 MiB
below the pooled worst; its distribution is 4022 / 4294 ×19 / 4296 ×4 / 4378 —
a spread of 356 MiB. W1 added 99 more on its own cut, worst 3186, spread 40 MiB.

So the 7055 remains **one anomalous flip in 320 observed**, not a rate. The
"different problem with a different fix" that §7a warned about is not present.

---

## 2. P2 — THE GATE COULD NOT RUN, AND THAT IS THE FINDING

§5.1b lists three legitimate outcomes. The real one was a **fourth**:
`--pp-solve-cut` raises before it ever reaches the gate. Found by executing it,
which no previous shift had done — RUNSHEET §4 verified the flag **parses**
(8/8, "by building the real parser"), and parsing is not dispatch.

| # | defect | status |
|---|---|---|
| 1 | `_handle_pp_solve_cut` calls `self._pp_cut_token_shares()`; **no such method exists**, on any branch. `AttributeError` before the gate. Pre-existing at `95e2e0eb0e`, from `7362073945`. | **fixed** `14c82e33ea`, red-first (2F 1P → 3P) |
| 2 | `_canonical()` strips `"nvidia"` but not `"geforce"`, so `CardLibrary.has()` is False for **every GeForce card NVML reports**, against a library that holds RTX 5090 / 4090 / 3080. | **fixed** `db2c34b928`, red-first (4F 2P → 3P + 3 subtests) |
| 3 | **Zero of the 16 seed cards carry measured `gemm_tflops`/`membw_gbs`**, and `_pp_cut_card_rates` builds `CardLibrary()` seed-only — it never calls the `CardLibrary.load(path)` classmethod that exists for exactly this. So the gate refuses on rate absence for every card on every rig. | **NOT fixed — it is a measurement pass, not a patch** |

Defect 3 is `#584`'s shape exactly, and it is why `#363` failed in the previous
window: **a planner path whose measurement inputs nothing in the tree produces.**
Inventing rates into `SEED_CARDS` would be precisely the "an unpriced term reads
as free memory" failure the flag's own help text refuses.

**WIRED VERDICT: `--pp-solve-cut` cannot produce a cut on this rig.** Recorded
verbatim in `m485/p2_wired.txt`.

### What the gate admits once the measurement exists

With the rates a previous shift **actually measured on these cards**
(`evidence-631/s50/gate_check.py`: 5090 231.97 TFLOPS / 1533.8 GB/s, 3080 65.57
/ 717.4) supplied through the library's own public constructor, and every step
after the rate lookup being the wired path (`m485/p2_with_rates.py`):

```
--pp-solve-cut .../m1/census: solved --pp-layer-ratio 37,14,13
    (attention per stage [9,3,4] of 16)
    makespan=100.6ms  pacer=stage1  min_headroom=129MiB
```

**Runsheet outcome (2): it admits a DIFFERENT cut.** Per §6a the **+25.5 % is
retired** and owed a re-measurement on `37,14,13` against a same-shift ship
control. The old number is not carried forward anywhere in this document.

### And the answer depends on which census feeds it

Fed the **M0 ship-cut** census instead of M1's, the same gate at the same pool:

```
--pp-solve-cut found no feasible layer split: ... the model needs 31218 MiB of
weights plus 8750 MiB of KV for a 280000-token arena, against 45117 MiB usable
across 3 ranks after the 1024 MiB corridor and per-rank transients.
```

Outcome (3) — *admits nothing* — from the same code, same pool, same budgets,
differing only in which boot's census supplied the calibration. Note the
aggregate is not the binding constraint (39968 needed vs 45117 usable): the
refusal is a **contiguity/packing** result.

**This is a circularity the ticket should name.** The gate calibrates per-layer
bytes, per-rank residual and the seam term from a census taken **under one
cut**, then chooses a different cut from that extrapolation. Two honest censuses
give opposite answers. Nothing here says which extrapolation is right; it says
the method has a free parameter that was not being reported.

---

## 3. W1 — CERTIFICATION WINDOW 1 ON THE ADMITTED CUT

`--pp-layer-ratio 37,14,13` (attention [9,3,4] derived), pool 280000,
`SGLANG_UNEVEN_TOKEN_VECTOR=9,3,4` following the attention split, flip ON,
23-minute soak, **291 requests, 0 errors**.

*(`--pp-attn-stage-ratio` cannot be passed alone — it steers the half that
`--pp-stage-ratio` derives — so the cut is spelled with `--pp-layer-ratio`
alone and the [9,3,4] split is the runtime's own derivation, confirmed by the
KV allocation 2.40 / 0.80 / 1.07 GB.)*

**Corridor, 100 ms NVML FREE, time-series minima over 16473 ticks per card:**

| card | role | minimum | samples < 1024 |
|---|---|---|---|
| gpu0 | rank 1, 3080 | 6537 MiB | **0** |
| gpu1 | **rank 0, 5090 — binding** | **3584 MiB** | **0** |
| gpu2 | rank 2, 3080 | 5671 MiB | **0** |

`CORRIDOR LAW BROKEN`: **0**. Tracebacks: **0**. Host axis: all 3 rank
processes survived, `oom_kill` delta **0**, MemAvailable min 42397 MiB.

**Both instruments, and they agree.** Seam-census per-state minima vs the 100 ms
sampler: rank 1 **6536.4 vs 6537**, rank 2 **5670.4 vs 5671** — under 1 MiB. On
the binding rank the census's finer trough is 3495.7 against the sampler's 3584,
i.e. the 100 ms sampler sits 88 MiB **above** the true minimum, which is the
only direction a coarser sampler may err.

**Verdict from `certify_485.py judge`:**

```
CLEAN  w1   gpu0=6537 gpu1=3584 gpu2=5671  binding=gpu1 margin=+2560 MiB
binding rank      gpu1, minima [3584]
margin            2560 MiB  (min minus the 1024 floor)
observed spread   0 MiB  over 1 window(s)

NOT CERTIFIED (C2): one window cannot bound a variance.
```

That is the correct and pre-registered answer, not a shortfall of this window.

**C2′ on the flip population** (the statistic the runsheet says is the evidence),
computed over W1's own reference class of 99 flips:

```
   at-rest baseline (binding)  6799 MiB
   worst transient              3186 MiB   (2nd worst 3146, spread 40 MiB)
   6799 - 3186 = 3613  >=  1024      PASSES by +2589 MiB
```

**Why W1's reference class is W1's own flips.** §6a establishes that these
numbers are properties of the cut. s50/s51 measured `40,12,12` and M1 measured
it again; W1 measures `37,14,13`. Charging W1 with s50's 7055 would be pooling
across configurations, which is the error §6a exists to prevent — and it is
also the reason the +25.5 % had to be retired. Stated explicitly because the
pooled number (`margin -1552` for M1 against the 221-flip pool) appears in the
evidence and must not be read as W1's.

**C4 — the gate must fund the seam.** The gate that admitted `37,14,13` was fed
`seam_staging_mib` from the M1 census's `SEAM_*` states; it did **not** fund
0.0, and `_pp_cut_seam_staging` now refuses a seam-less census rather than
defaulting. C4 is satisfied *on the seam term*. It is **not** satisfied as a
wired admission: defect 3 above means no boot can reproduce that admission
without borrowed rates. **C4 is not closed.**

**C5 — the census must see the cutover.** Satisfied: all three ranks recorded
`SEAM_PP_TO_TP` and `SEAM_TP_TO_PP` with **99 samples each**.

**One caveat this window carries.** W1 booted with `PHASE-FLIP-SEAM-RESERVE ...
is COLD` on all three ranks — no cached seam record exists for a cut never
booted before — so it sized with **no flip-seam term** and still held 2560 MiB.
A warm-reserve boot of the same cut is a different (and easier) configuration;
this result is the cold one.

---

## 4. THE TWO INSTRUMENTS DISAGREED BY 3750 MiB, AND THE RECONCILIATION IS EXACT

Worth recording because it looks like a defect and is not. On M1 rank 0 the
transient census reports `SEAM_TP_TO_PP = 544 MiB` while the seam census reports
a modal draw of **4294 MiB** for the same flips.

They agree on the **trough**, which is the physical quantity:

```
   seam census   worst trough, rank0 tp_to_pp   2147     MiB
   transient census  min_free[SEAM_TP_TO_PP]    2147.6875 MiB
```

**0.7 MiB apart** — the same agreement window 1 showed. The difference is the
BASELINE each differences against, and both are documented choices: the seam
census uses free at cutover ENTRY (6527), the transient census uses its own
AT-REST baseline (2691.7), because every other state in its table is
denominated that way. `note_free()`'s docstring says exactly this. The gate
reads the transient census, and for the gate the at-rest-relative term is the
self-consistent one, since residency is priced from the same at-rest census.

---

## 5. WHAT THE NEXT SHIFT OWES

1. **Defect 3 — a card-rate measurement pass** (`#584`). Until it exists,
   `--pp-solve-cut` is unusable on every rig, C4 cannot close, and the
   `37,14,13` admission rests on borrowed numbers. Wiring
   `CardLibrary.load(path)` into `_pp_cut_card_rates` is the smaller half; the
   probe that fills the file is the real work.
2. **`certify_485.py` is not on the line.** The MERGED runsheet's §7a and §9
   depend on it; it exists only on `chore/ticket-485-cert`. This shift ran the
   judge out of `/spinning/wt-485-cert`. Merge it, or the ticket cites a tool
   the tree does not have.
3. **W2, W3 on `37,14,13`.** C2 needs a spread, and every window narrows the
   right tail. W1's own class is 99 flips at spread 40 MiB — encouraging, and
   one boot.
4. **Re-measure the speed on the admitted cut** against a same-shift ship
   control (§6a). The +25.5 % is retired and nothing replaces it yet. The A/B
   depth probe was deliberately not run here, per §7a's "do NOT do" list.
5. **Name the census-dependence of the solve** (§2) in the runsheet. Two
   censuses, opposite verdicts, is a property of the method that currently goes
   unreported.
