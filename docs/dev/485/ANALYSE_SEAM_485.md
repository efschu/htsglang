# SEAM ANALYSIS — #485, can the seam be reduced far enough to certify a cut?

**Question this answers.** `WINDOW_VERDICT_584_M584.md` §5 closed the #485
planner cut with a REFUSAL and named exactly one recovering lever: *"the seam
must fall to ≈515 MiB — a reduction of ≈1323 MiB, 72 % of the measured draw"*.
This is the feasibility verdict on that lever: what the seam transient is made
of, which mechanisms can attack which term, and whether the threshold is
reachable.

**Derived from** the merged line `feat/route-a-631` == `integration/r2` @
`ac0d1f36f3`. Branch `feat/desk-seam-485`, worktree
`/spinning/wt-desk-seam-485`. **Desk only** — no GPU window claimed, no serving
process touched, no boot. The solver runs are CPU arithmetic over a census on
disk.

**Reproduce every number below:**

```bash
cd /spinning/wt-desk-seam-485
# the decomposition
python3 scripts/seam_485/seam_decompose_485.py \
  m0=/spinning/evidence-631/m485/m0/boot.log \
  m1=/spinning/evidence-631/m485/m1/boot.log \
  w1=/spinning/evidence-631/m485/w1/boot.log
python3 scripts/seam_485/seam_terms_485.py \
  /spinning/evidence-631/m485/m0/census /spinning/evidence-631/m485/m0/boot.log
# what admission actually requires
/spinning/htsglang-gpu/.venv/bin/python3 \
  scripts/seam_485/seam_target_485.py probe \
  /spinning/evidence-631/m485/m0/census /tmp/seam485_probe
```

---

## 0. VERDICT, first

**NOT REACHABLE at the certification pool of 280000.** The best-achievable seam
vector — every mechanism in §5 working perfectly, simultaneously, on every
rank — still refuses, and it refuses by 170 MiB against a physical floor that
this fork's own source already documents.

```
  best achievable at pool 280000, all mechanisms perfect:
      rank0 SEAM_TP_TO_PP = 716 MiB   (169 entry deficit + 547 one-layer floor)
      every other seam term = 0
  ->  REFUSE

  admission ceiling on rank0 SEAM_TP_TO_PP, bisected on the merged line
  with all five other terms held at exactly zero:
      546 MiB
  ->  short by 170 MiB
```

**The threshold IS reachable — but only by moving the pool, which #584 recorded
as not a lever.** The same best-achievable vector admits at pool 200000 and
below:

| pool | best-achievable rank0 `SEAM_TP_TO_PP` | one-layer floor | verdict |
|---:|---:|---:|---|
| 280000 | 716 | 547 | **refuse** |
| 200000 | 560 | 391 | **ADMIT** `29,19,16` |
| 160000 | 481 | 312 | ADMIT |
| 120000 | 403 | 234 | ADMIT |

So the honest answer to *"is ≤515 MiB reachable"* is: **no on rank 0 at the
certification pool, and the only combination that admits is `chunked refill +
wave-at-its-floor + a ≥29 % pool reduction`** — a mechanism stack that is
L-sized and unbuilt, paid for with a context-capacity cut.

**Three findings overturn the premise the lever was stated on.**

1. **The 1838 MiB is not the arena.** On the governing M0 census the binding
   rank's `weights_refill` step is **+62 MiB — a release**, not −4150. `alloc`
   is flat to within 43 MiB across the entire flip while `free` swings 1670
   MiB, so the draw is outside torch (S3, settled — §2). The −4150 figure
   belongs to the M1 *arm* cut and does not describe the census that governs.
2. **It is not one number.** Setting rank 0's seam to **zero** while leaving
   ranks 1 and 2 at their measured values still **REFUSES** (§4). Five of the
   six seam terms are individually binding. The ask is a joint reduction across
   a six-coordinate vector, not 1838 → 515 on one rank.
3. **The census is off-configuration, and the seam is pool-linear.** M0 was
   measured at pool 620000; the gate certifies at pool 280000. The wave term
   scales with the pool (proven, §6), so the gate reads a seam **2.21× larger
   than the operating point it is deciding**. That mis-scaling is 915 MiB of
   #584's 1323 MiB ask — and correcting it still refuses, because what remains
   is the physical floor.

---

## 1. THE INSTRUMENT, and what its two numbers mean

Two instruments report a "seam", they do not report the same quantity, and the
difference is itself a load-bearing term.

| | produced by | quantity |
|---|---|---|
| seam census `transient` | `phase_flip_seam_census.py:337` | `entry_free − trough`, per flip |
| planner `SEAM_*` term | `transient_census.py:292`, fed the seam census's trough | `baseline_free_mib − worst_trough`, per window |

The planner's term — the one `_pp_cut_seam_staging` reads and the one #584
swept — is therefore **the flip's own draw plus however far the rank already
sat below its census baseline when the flip began**. That second part is real,
it is measured, and no seam mechanism touches it. It is carried below as
`entry_deficit`.

The census line carries `free`, `slack`, `alloc` and `res` at every stage mark
(the T-ticket landed in `EXCURSION_ANALYSIS_485.md` §8). `alloc` is the
decisive column: **flat while `free` drops means the bytes went outside torch**
(`phase_flip_seam_census.py:307-317`).

**Population.** 1014 `alloc`-bearing seam-census lines over the three merged
windows = **169 flips**, each reported on 3 ranks × 2 directions (M0 32, M1 38,
W1 99). Larger seam-census populations exist on disk from earlier shifts but
predate the `alloc=` field and so cannot answer the S1/S3 discriminator; they
are not pooled here.

| window | cut (of 64 layers) | pool | role |
|---|---|---:|---|
| **M0** | 28,20,16 (ratio 14,10,8) | 620000 | **the governing census** — §6b takes the most demanding |
| M1 | 40,12,12 | 280000 | arm |
| W1 | 37,14,13 | 280000 | arm |

---

## 2. THE DECOMPOSITION — per term, at the instant of the trough

Governing census M0. Each row is the flip that *sets* that window's `SEAM_*`
number (the worst-trough flip). Signs as measured: negative = drawn and still
held at the trough.

| rank / leg | census | at_rest | entry_def | wave | kv_stage | arena | gdn | other | Δalloc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 pp_to_tp | 1302 | 3954 | 317 | **−1986** | −6 | 0 | 0 | +1006 | 0 |
| **0 tp_to_pp** | **1838** | 3954 | 169 | **−1392** | −278 | **0** | 0 | 0 | +43 |
| 1 pp_to_tp | 164 | 1680 | 164 | −576 | −8 | 0 | −94 | +738 | 0 |
| 1 tp_to_pp | 490 | 1680 | −458 | −152 | −298 | **−594** | 0 | +96 | +29 |
| 2 pp_to_tp | 256 | 2074 | 256 | −560 | −210 | 0 | −128 | +1078 | +7 |
| 2 tp_to_pp | 408 | 2074 | −1420 | −96 | −230 | **−1564** | 0 | +62 | +17 |

`census = entry_deficit − Σ(held terms)`; it closes to the MiB on rank 0 (169 +
1392 + 278 = 1839 against a reported 1838) and to a few tens of MiB on ranks
1–2, where the worst-trough flip and the baseline-setting flip are not the same
flip.

**Δalloc ≤ 43 MiB on every row while `free` moves by up to 1986 MiB.** This is
the discriminator `EXCURSION_ANALYSIS_485.md` §4 asked one window to decide,
and the answer is **S3 across the board**: the seam draw is VMM commit traffic
outside torch, not a caching-allocator excursion. That question is now closed
from artifacts already on disk; no window is owed for it.

### 2b. The terms are deterministic, not distributional

Flip-to-flip spread over all 32 M0 flips, load-bearing terms only:

| term | min | p50 | max |
|---|---:|---:|---:|
| rank 0 tp_to_pp `wave` | −1412 | **−1392** | −1392 |
| rank 0 tp_to_pp `kv_stage` | −278 | −278 | 0 |
| rank 1 tp_to_pp `arena` | −594 | **−594** | −594 |
| rank 2 tp_to_pp `arena` | −1564 | **−1564** | −1564 |

A 20 MiB spread over 32 flips on the binding term. **The seam is an identity,
not a random variable** — which is what makes a desk verdict possible, and what
makes "run another window and hope" a non-strategy.

### 2c. The arena term is an identity, and it is small where it matters

From the rung-3 log lines both layout sizes are recoverable per rank, and

```
  arena_tail(rank) = | layout_PP.total_bytes − layout_TP.total_bytes |
                     + 128 MiB checksum, on the leg that GROWS
```

committed by `_commit_refill_high_water()` (`phase_flip_boot.py:464-470`) to
`max(both layouts)`, released again by `set_active_prefix(active_layout)`:

| rank | TP layout | PP layout | Δ = tail | measured `weights_refill` step |
|---|---:|---:|---:|---|
| 0 (M0, 28L) | 13672 | 13482 | **190** | tp→pp **+62** = +190 release − 128 checksum ✓ |
| 1 (M0, 20L) | 7659 | 8125 | 466 | tp→pp −594 = −466 − 128 ✓ |
| 2 (M0, 16L) | 7659 | 9095 | 1436 | tp→pp −1564 = −1436 − 128 ✓ |
| 0 (M1, 40L) | 13692 | 17842 | 4150 | tp→pp −4278 = −4150 − 128 ✓ |
| 0 (W1, 37L) | 13692 | 16754 | 3062 | tp→pp −3190 = −3062 − 128 ✓ |

The PP layout is linear in the rank's layer count — **363.3 MiB/layer on rank 0**,
366.3 on rank 1 — so the tail at any candidate cut is computable without a
boot. **At the candidate cut `29,19,16` the tails are 154 / 152 / 1436 MiB.**

This is why the brief's premise — *"refill GROWS the arena via real cuMemCreate
~4150 MiB per flip"* — is true of M1 and false of the census that governs.
Rank 0's two layouts differ by 190 MiB under the ship cut. **An arena mechanism
recovers essentially nothing on the binding rank.**

### 2d. The wave is the binding rank's whole seam

Splitting the wave into what it *holds at the trough* versus what it *keeps at
the end of the flip*:

| rank / leg | held at trough | net over the flip | apparent coexistence |
|---|---:|---:|---:|
| **0 pp_to_tp** | −1986 | −834 | **1152** |
| **0 tp_to_pp** | −1392 | −264 | **1128** |
| 1 pp_to_tp | −576 | −576 | **0** |
| 1 tp_to_pp | −152 | −152 | **0** |
| 2 pp_to_tp | −560 | −560 | **0** |
| 2 tp_to_pp | −96 | −96 | **0** |

**Rank 0 is the only rank with any removable wave coexistence at all.** On
ranks 1 and 2 the wave is entirely a persistent residency step into the other
phase's layout — held equals net, so there is nothing for a scheduling
mechanism to recover. That kills mechanisms (c) and (d) on those ranks before
they are evaluated.

The 1128/1152 column is labelled *apparent* deliberately: §5(d) shows most of
it is below a physical floor and is not in fact removable.

---

## 3. WHERE THE TROUGH ACTUALLY IS

The census names the trough stage per flip, and it moves with the cut:

| window | rank 0 tp_to_pp trough stage |
|---|---|
| **M0 (governing)** | `kv_pack` 30/32, `backing_restore_span` 2/32 — **`weights_refill` 0/32** |
| M1 | `weights_refill` 38/38 |
| W1 | `weights_refill` 99/99 |

On the governing census the trough is in the **KV backing wave**, and the
weights refill happens ~1200 MiB *above* it and *releases*. **The stage the
prior analysis is organised around is not the stage that binds the cut being
certified.**

The M0 trace shows why: the wave is a sawtooth of restore-heavy groups (16
restore/release pairs, net −696) alternating with release-only groups (+512).
The trough is set by the **first two restore-heavy groups running back to
back** before any release-only group; every later local minimum is shallower
(2443, 2771, 2587, 2403, 2219 against the trough's 2115).

---

## 4. WHAT ADMISSION REQUIRES — it is a vector, not a number

`seam_target_485.py` sets each `SEAM_*` entry to an absolute MiB value
independently, so a verdict can be attributed to the term a mechanism would
actually touch. **It reproduces #584's uniform sweep exactly on the merged
line** (x0.35 refuse / x0.25 ADMIT `29,19,16` / x0.0 ADMIT), so the divergences
below are the probe's, not the harness's.

| probe | result |
|---|---|
| baseline, as measured | refuse |
| **rank 0 both legs → 0, ranks 1/2 at measured** | **refuse** |
| rank 0 → 0, ranks 1/2 ×0.50 / ×0.45 / ×0.40 | refuse |
| rank 0 → 0, ranks 1/2 ×0.35 | ADMIT |
| ranks 1/2 → 0, rank 0 ×0.50 / ×0.35 | refuse |
| ranks 1/2 → 0, rank 0 ×0.25 | ADMIT |

Coordinate probe from the x0.25 admit point, restoring one term to measured:

| term restored | verdict |
|---|---|
| rank 0 `SEAM_TP_TO_PP` → 1838 | refuse |
| rank 0 `SEAM_PP_TO_TP` → 1302 | refuse |
| rank 1 `SEAM_TP_TO_PP` → 490 | refuse |
| rank 1 `SEAM_PP_TO_TP` → 164 | **ADMIT** — the only slack term |
| rank 2 `SEAM_PP_TO_TP` → 256 | refuse |
| rank 2 `SEAM_TP_TO_PP` → 408 | refuse |

**Five of six terms are individually binding.** #584's single-number framing is
the projection of a six-dimensional constraint onto its largest coordinate, and
it materially understates the ask: a mechanism that perfectly solves the
binding rank still refuses.

### 4b. Individual ceilings, bisected

Each term swept with **every other term held at exactly zero** — the most
generous configuration that exists — at the certification pool of 280000:

| term | measured | ceiling | cut needed |
|---|---:|---:|---:|
| rank 0 `SEAM_TP_TO_PP` | 1838 | **546** | 1292 |
| rank 0 `SEAM_PP_TO_TP` | 1302 | **546** | 756 |
| rank 1 `SEAM_TP_TO_PP` | 490 | **327** | 163 |
| rank 1 `SEAM_PP_TO_TP` | 164 | **327** | −163 (slack) |
| rank 2 `SEAM_TP_TO_PP` | 408 | **156** | 252 |
| rank 2 `SEAM_PP_TO_TP` | 256 | **156** | 100 |

These are upper bounds on what any mechanism may leave behind, and they are not
simultaneously available — the joint constraint of §4 is tighter than any of
them. Two readings matter beyond the arithmetic:

* **Rank 1 `SEAM_PP_TO_TP` is the only term with slack**, 163 MiB of it, which
  is exactly what the §4 coordinate probe found independently.
* **Rank 2 carries the tightest ceiling in the system at 156 MiB** — tighter
  than the 5090's 546 — because it is a 20 GB card holding 16 layers *and* the
  `lm_head`. Any mechanism proposal must clear rank 2, and the only term large
  enough to matter there is its 1436 MiB arena tail, i.e. mechanism (b).

---

## 5. MECHANISM VERDICTS

### (a) PERSISTENT ARENA — keep the tail mapped across flips

The flag already exists: `KvVmmArena(retain_handles=True)` parks the physical
handle on decommit instead of releasing it, so the next same-size commit reuses
it with no driver round trip (`kv_vmm_backing.py:350, 402-433`; the class
docstring calls it the "#631 ZERO-ALLOCATION SEAM"). It is off for the
phase-flip weights carrier (`phase_flip_spill.py:912, 916`). **Size: S — one
constructor argument.**

**MiB recovered:** the `arena` term only, and only on the growing leg. At the
candidate cut `29,19,16`: rank 0 **154**, rank 1 **152**, rank 2 **1436**.
Against rank 0's ask of 1292 MiB (§4b) this covers **12 %**; on the governing
census as measured it covers **0**.

**At-rest cost:** the same MiB, resident in *both* phases instead of one.
Charged against M0's measured at-rest free:

| rank | at-rest free | tail retained | at-rest free after | corridor (1024) |
|---|---:|---:|---:|---|
| 0 | 3954 | 190 | 3764 | ok |
| 1 | 1680 | 466 | 1214 | ok, 190 MiB of margin |
| 2 | 2074 | 1436 | **638** | **BROKEN** |

**Correctness risk:** low. The VA is boot-reserved and never rebound
(`phase_flip_spill.py:839-896`); captured graphs hold arena addresses that do
not move and `contains_all_params()` (`phase_flip_boot.py:884-911`) still
holds. Retention changes only whether the physical page is returned.

**Verdict: REJECTED.** It drives rank 2's at-rest free to 638 MiB, **breaking
the corridor law at rest** — trading a transient for a permanent breach is not
a reduction. The solver refuses it independently: seam minus the arena term on
every rank → refuse; the same with the at-rest cost charged to
`--rank-gpu-memory-mib` → refuse.

> **The register lesson applies here, and it cuts against the mechanism.**
> Retention changes the frequency from once-per-flip to once-per-process, so
> its cost must be re-priced at the new frequency — and at once-per-process it
> is paid in the phase that does not need it, for every second the instance is
> up, instead of for the ~280 ms of a seam.

### (b) INCREMENTAL / CHUNKED REFILL — commit the high-water in slices

Same target term as (a), **at zero at-rest cost**: commit a slice, copy it,
release it, repeat, so peak coexistence is one slice rather than the whole
|PP − TP| delta. The arena already supports the primitives —
`commit_span`/`decommit_span` (`kv_vmm_backing.py:733, 804`) and a
`commit_chunk_bytes` granule.

**MiB recovered:** 154 / 152 / 1436 at the candidate cut, at-rest cost 0.
Strictly better than (a) and it is what (a) should have been.

**Correctness risk — the reason it is not already done.**
`phase_flip_boot.py:373-379, 405-412` documents the contract: `arena_refill`'s
`restore=` arm can rewrite the *other* layout in place on a checksum mismatch,
so **both layouts must be backed before any byte moves**, or the recovery path
faults on unbacked memory. Slicing breaks that atomicity. A correct version
needs either per-slice checksums with per-slice recovery, or an explicit
demotion of the recovery guarantee — a design decision, not a refactor.

**Size: M.** **Verdict: SOUND, NECESSARY, INSUFFICIENT.** It is in every
admitting combination in §0, and it is worth building on its own merits (it
removes a real 1436 MiB transient on rank 2 at no at-rest cost). It does not by
itself reach any threshold.

### (c) STAGED OVERLAP REDUCTION — stop the wave peak and the arena coexisting

The gate is additive today (`phase_flip_runtime.py:4122-4125`,
`arena_tail + max(wave_peak, draft_restore)`), and R11's comment justifies the
additivity: `stacks.refill` is a *pre*-cutover function, so it commits while
the wave state is still outstanding. If the two could be made sequential the
reserve would drop by `min(term)` rather than staying at the sum.

**MiB recovered:** on rank 0 of the governing census the arena term is **0**,
so there is nothing to de-overlap on the binding rank — `min(1392, 0) = 0`. On
ranks 1 and 2 the wave held at the trough is 152 and 96 MiB against arena terms
of 594 and 1564, so sequencing recovers `min` = **152 and 96 MiB**, on the
ranks where it helps least.

**Size: M.** **Verdict: NOT A LEVER.** The two terms the additive gate prices
as coexisting are, on the census that governs, on *different ranks*. The
premise that ordering can help is a property of the M1 arm cut.

### (d) WAVE EXCLUSIVITY — the mechanism the decomposition names

Rank 0's entire seam is the wave, so this is the only mechanism that addresses
the binding term. **It already exists, and it was already on in every measured
window.**

`_stream_wave` (`phase_flip_runtime.py:4373-4450`) restores, writes and
releases **one row block at a time** inside a wave, driven by
`SGLANG_FLIP_SEAM_ROW_BLOCKS` (default 16) and gated on
`SGLANG_FLIP_SEAM_CHUNK_MIB`, which `ship_env_live.txt` sets to **8** and which
`boot_m.sh` replays into all three windows. The `_span` stage names in the
census (`backing_restore_span` / `backing_release_span`) are emitted only by
that path, so **the measured 1392 MiB is already the row-block-streamed
number**, not the naive one.

The obvious inversion is already known-worse. Release-before-restore exists
(`SGLANG_FLIP_SEAM_RESTORE_FIRST=0`), and `phase_flip_runtime.py:5756-5786`
records why it was abandoned: release-first makes a wave's releases pay for its
own commits, which caps the wave count at the smallest PP stage and *raises*
the staging slope. For aliased pools restore-first would be outright wrong
(`memory_pool.py:2591-2601`), so the order is correctness-gated, not free.

**The floor, from this fork's own source** (`memory_pool.py:2587-2600`,
verbatim):

> *"…leaves an irreducible transient of ONE destination layer. In the tp_to_pp
> direction that layer spans the FULL pool (a PP stage holds all tokens of its
> layers), i.e. **1.953 MiB per 1000 pool tokens**, and no ordering removes it:
> a peer cannot release a source layer until its owner has written it, and the
> owner cannot write until its destination layer is backed."*

**That constant is confirmed twice by the solver itself.** The model has 16
full-attention layers: `16 × 1.953 × 280 = 8749.4 MiB` against the solver's
*"8750 MiB of KV for a 280000-token arena"*, and `16 × 1.953 × 40 = 1250.0`
against its *"1250 MiB of KV for a 40000-token arena"*. The floor is not a
comment's claim; it reconciles with the gate's own KV term to 0.6 MiB.

| pool | one-layer floor | measured / scaled held wave (rank 0 tp_to_pp) | actually removable |
|---:|---:|---:|---:|
| 620000 (M0) | **1211** | 1392 measured | **181 MiB (13 %)** |
| 280000 (cert) | **546.8** | 629 scaled | **82 MiB (13 %)** |

**Size: L.** **Verdict: ALREADY DEPLOYED, AND WITHIN 13 % OF ITS FLOOR.** The
1128 MiB of "coexistence" in §2d is an upper bound that ignores physics; the
genuinely removable part is 181 MiB at the measured pool and 82 MiB at the
certification pool. The one unmeasured knob left is
`SGLANG_FLIP_SEAM_ROW_BLOCKS > 16` against an 8 MiB commit chunk — the code's
own note says B=32 reproduces B=16 *when the chunk is 16 MiB*, so at chunk 8
there may be one more step. It cannot be worth more than the 82 MiB the floor
leaves.

### (e) Mechanisms the decomposition suggests and also kills

* **Attack `entry_deficit`** (169 MiB on rank 0 tp_to_pp, 317 on pp_to_tp) —
  the rank sitting below its own census baseline at flip entry. Not a seam term
  at all; no flip mechanism reaches it, and mechanism (a) makes it *worse*.
* **Attack `kv_stage`** (278 MiB on rank 0 tp_to_pp, spread −278/−278/0). The
  `kv_pack` / `kv_local_read` staging buffers; worth 126 MiB at the
  certification pool. Assumed fully removable in every "best achievable" figure
  in this document, which is generous. Same reads-before-writes hazard region
  (`phase_flip_runtime.py:5735`).

---

## 6. THE POOL — the register lesson, and #584's L3 verdict corrected

**The seam term is pool-linear, and this is measured, not modelled.** The same
term on the same rank across two pools:

| rank 0 pp_to_tp held wave | pool | ratio to M0 |
|---:|---:|---:|
| 1986 MiB | 620000 | 1.00 |
| 908 MiB (M1) | 280000 | 2.19 |
| 804 MiB (W1) | 280000 | 2.47 |

against a pool ratio of 620/280 = **2.214**. The wave is KV backing; it scales
with the KV pool, as it must.

**The governing census was measured at pool 620000. The gate certifies at pool
280000.** So `_pp_cut_seam_staging` reads a seam term **2.21× larger than the
operating point it decides**, and 915 MiB of #584's 1323 MiB ask is pure unit
mismatch. Correcting for it, and for the candidate cut's own arena tails (§2c):

| | r0 tp | r0 pp | r1 tp | r1 pp | r2 tp | r2 pp | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| as fed to the gate today | 1838 | 1302 | 490 | 164 | 408 | 256 | refuse |
| (a)/(b) arena removed on the census as it stands | 1838 | 1302 | 0 | 164 | 0 | 256 | refuse |
| **E1** re-derived at pool 280000, cut 29,19,16 | 1204 | 210 | 243 | 0 | 229 | 0 | **refuse** |
| **E2** + chunked refill (b) removes the arena | 922 | 210 | 0 | 0 | 0 | 0 | **refuse** |
| **E3** + wave at its documented floor — **best achievable** | **716** | 210 | 0 | 0 | 0 | 0 | **refuse** |
| E4 + rank 0's *apparent* coexistence removed — **not physical** | 414 | 214 | 0 | 0 | 0 | 0 | ADMIT |

**E3 is the verdict: the best physically achievable vector refuses**, at 716
MiB against a ceiling of 546 (§4b), short by 170 MiB.

**E4 admits and E4 is not physical.** It puts rank 0's `SEAM_TP_TO_PP` at
414 MiB where the irreducible one-layer transient at that pool is 546.8 MiB.
The distance between E3 and E4 is the arithmetic proof that the admissible set
and the achievable set do not overlap at this pool: the whole gap between them
falls inside the floor.

### 6b. #584's L3 conclusion rests on an instrument defect — and the correction changes it

#584 swept `--max-total-tokens` 280000 → 20000, recorded every value as
refused, and concluded the pool *"CONFIRMED NOT A LEVER"*. That sweep changed
the pool in the solve **while holding the census's seam terms at their
620000-measured values** — i.e. it was measured on a model in which the pool
has no effect on the seam, which §6 shows is false.

Re-swept with the seam coupled to the pool it is measured against, and with the
mechanism stack applied:

| pool | best-achievable r0 tp | one-layer floor | verdict |
|---:|---:|---:|---|
| 280000 | 716 | 547 | refuse |
| 200000 | 560 | 391 | **ADMIT** `29,19,16` |
| 160000 | 481 | 312 | ADMIT |
| 120000 | 403 | 234 | ADMIT |

**The pool IS the enabling lever, once it is allowed to move the seam too.**
The ceiling relaxes with falling pool faster than the floor falls, so a gap
opens somewhere between 200000 and 280000 tokens.

Two caveats stated plainly.

* The admitting vectors above assume **every** mechanism perfect
  simultaneously: chunked refill removing the whole arena term on all ranks,
  the wave sitting exactly on its documented floor, and `kv_stage` fully
  eliminated. None of that is built; (b) is M-sized and (d) is already within
  13 % of its floor.
* Reducing the pool from 280000 to 200000 is a **29 % context-capacity cut**,
  which runs directly against the MAX-KV requirement the flip-setup capacity
  spec pins for bs1. Whether that trade is acceptable is a product decision,
  not a desk one, and it is not made here.

Separately, the raw coupled sweep *without* the mechanism stack refuses at
every pool from 280000 down to 40000, and at 40000 the binding coordinate is
rank 2 — `(276,128,69,38,103,50)` refuses while the same vector with rank 2
zeroed admits. Rank 2's packing margin is under ~100 MiB, so any mechanism
proposal must clear rank 2 as well as rank 0.

---

## 7. WHAT WOULD CHANGE THIS VERDICT

Stated so a future shift does not re-run what is already closed.

1. **Not another window.** §2b: the binding terms have a 20 MiB spread over 32
   flips. More samples narrow nothing, and the S1/S3 question the last ticket
   was written for is answered in §2 from artifacts already on disk.
2. **Not the budget.** Closed by #584 §5 L1 — refused at full nameplate on
   every card; it is a packing result.
3. **Not a seam mechanism alone.** (a) is corridor-illegal on rank 2, (b) is
   sound but lands 1436 MiB on the wrong rank, (c) has nothing to de-overlap on
   the binding rank, (d) is deployed and within 13 % of a physical floor.
4. **The pool, coupled with (b) + (d).** The only combination that admits.
   Costs ≥29 % of context. §6b.
5. **A different KV geometry moves the floor itself.** The 1.953 MiB/1000-token
   constant is `one full-attention layer × full pool`. Anything that makes a PP
   stage's destination layer smaller than the whole pool — a chunked or paged
   destination backing, or a PP stage that does not hold all tokens of its
   layers — attacks the floor rather than the scheduling around it. That is a
   change to the exclusive-backing contract, and it is the only listed route
   that neither costs context nor is already closed.

**Recommended disposition.** Record #485's planner cut as **REFUSED at pool
280000, with a named and costed route to admission** rather than as a bare
refusal. Build (b) chunked refill on its own merits. Do not re-run the seam
lever as a single-number target, and do not book a certification window until
either the pool decision (item 4) or the geometry change (item 5) is taken.

---

## 8. DESK VALIDATION PERFORMED

| item | result |
|---|---|
| harness reproduces #584's published sweep on the merged line | x1.0 refuse / x0.35 refuse / **x0.25 ADMIT `29,19,16`** / x0.1 ADMIT / x0.0 ADMIT — identical to `m584/p2_seam_lever_m0.txt` |
| census parser reads both the pre- and post-`alloc=` formats | 1014 lines over 3 windows, 169 flips, 0 unparsed |
| decomposition identity closes | rank 0 tp_to_pp: 169 + 1392 + 278 = 1839 against a reported 1838 |
| arena identity closes against the rung-3 log | 5/5 rows exact (§2c) |
| 1.953 MiB/1000-token floor cross-checked against the solver's KV term | 8749.4 vs *"8750 MiB"* at pool 280000; 1250.0 vs *"1250 MiB"* at pool 40000 |
| pool-linearity of the wave | 2.19 and 2.47 measured against a pool ratio of 2.214 |
| `alloc` discriminator (S1/S2 vs S3) | Δalloc ≤ 43 MiB on all six rows — **S3**, settled |
| ceiling bisection, all six terms | rank 0 both legs **546**, rank 1 both legs **327**, rank 2 both legs **156** |
| `seam_feasibility_485.py mechanisms` reproduces §6's E-table | 5/5 rows, E3 best-achievable `(716, 210, 0, 0, 0, 0)` **refuse** |
| `seam_feasibility_485.py pools` reproduces §6b | refuse at 280000, ADMIT at 200000 / 160000 / 120000 / 80000 |
| `ruff check scripts/seam_485/` | clean |

No GPU, no serving process, no model touched, no boot, no `git stash`.
