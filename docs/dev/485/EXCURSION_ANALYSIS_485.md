# EXCURSION ANALYSIS — #485, the 686 MiB on the binding rank

**Question this answers.** `RUNSHEET_485_CERTIFICATION.md` §2 resolved that s50's
corridor breach to 669 MiB stays in the reference class, and §2 option **(b)**
was chosen: *explain* the excursion rather than out-vote it with more windows.
This is that explanation.

**Derived from** `integration/r2` @ `2b71b5b242` (re-resolved 2026-08-14).
Branch `feat/desk-485-excursion`, worktree `/spinning/wt-desk-485`.
CPU only — no GPU window claimed, no serving process touched, no boot.

**Reproduce every number below in three commands:**

```bash
cd /spinning/wt-desk-485
python3 scripts/cert_485/excursion_485.py census \
  --log s50=/spinning/evidence-631/s50/boot_gate_cut_280000.log \
  --log s51=/spinning/evidence-631/s51/boot_arm_280000.log
python3 scripts/cert_485/excursion_485.py decompose \
  --log /spinning/evidence-631/s50/boot_gate_cut_280000.log --at 11:44:06
python3 scripts/cert_485/excursion_485.py judge \
  --window s50=/spinning/evidence-631/s50/boot_gate_cut_280000.log \
  --window s51=/spinning/evidence-631/s51/boot_arm_280000.log
```

---

## 0. VERDICT, first

**The mechanism is named, not merely suspected.** The excursion is one
anomalous `tp_to_pp` phase-flip seam on rank 0, whose `weights_refill` stage
drew **5536 MiB instead of the invariant 4278 MiB** — an excess of **1258 MiB**.
Every other stage of that flip is byte-identical to the modal flip, and the
code path is identical (mechanical diff, §3).

**But the headline finding is bigger than the outlier, and it changes the
certification.** The corridor minimum on the binding rank is not a random
variable. It is an identity, and every term is measured:

```
  min_free(rank 0) = at_rest_TP_free - wave_residual - arena_tail - checksum
   s50 modal:  1925 = 7725 - 1522 - 4150 - 128        (transient 5800)
   s51 modal:  1356 = 7314 - 1680 - 4150 - 128        (transient 5958)
   s50 breach:  668 = 7723 - 1522 - 4150 - 128 - 1258 (transient 7055)
```

The gate funds **4148.35 MiB** of that — the arena tail alone. The planner's
`seam_staging_mib` funds **0.0**. So the 331 MiB of "margin" the runsheet
certified against was never a budgeted quantity; it is the leftover after an
unfunded 5800 MiB draw, and the 686 MiB "spread" is the difference between two
leftovers.

**Consequence.** Neither window ever had the margin. Under the amended
criterion (§7) s50 fails by 354 MiB and s51 fails by 765 MiB. **s51 was not the
lucky boot — it was the boot with the *thinner* routine margin (332 MiB against
s50's 901 MiB) that simply never drew the outlier.** Had the 7055 MiB seam
occurred during s51, its trough would have been **259 MiB**.

---

## 1. The instrument that already answers this

The fork instruments every seam. One line per flip per rank, on by default:

```
[#631 seam-census] tp_to_pp rank 0: transient 5800 MiB
  (baseline free 7725 MiB, trough 1925 MiB at 'weights_refill')
  | plan free=7725 step+0 slack=729 | kv_pack free=... | ... (480 stage marks)
```

`phase_flip_seam_census.py` samples `(free, reserved, allocated)` at each stage
mark on the cutover path. **The distribution over a whole window is the
finding**, and no one had computed it — a 480-mark line is not something a
human reads off a 19 MB log.

Both windows carry it. Pooled: **196 `tp_to_pp` flips on rank 0.**

| | s50 (`boot_gate_cut_280000.log`) | s51 (`boot_arm_280000.log`) |
|---|---:|---:|
| `tp_to_pp` flips on rank 0 | 86 | 110 |
| transient — modal | **5800** MiB | **5958** MiB |
| transient — body range | 5654 – 6008 | 5792 – 5998 |
| transient — **worst** | **7055** | 5998 |
| baseline free — modal | 7725 | 7314 |
| trough — modal | 1925 | 1356 |
| trough — worst | **668** | 1354 |
| corridor breaches | 1 | 0 |

**Ranks 1 and 2 report `transient 0` on all 86 flips.** That is the structural
reason gpu0 reproduces to 6 MiB and gpu2 to 50 MiB while gpu1 moves: only the
binding rank carries a seam transient at all. The reproducibility contrast the
runsheet treated as evidence of a rank-specific *noise* source is evidence of a
rank-specific *mechanism*: the arena tail is committed only where the PP layout
is the larger one, which under the `40,12,12` cut is rank 0 alone.

---

## 2. The transient is quantized, and the quanta are identifiable

The step charged to `weights_refill` takes exactly **three** values in 196
flips across two boots:

| step | s50 | s51 | what it is |
|---:|---:|---:|---|
| **−4150 MiB** | 6 | 9 | the arena tail alone (`staging reserved 4148.35 MiB`) |
| **−4278 MiB** | 79 | 101 | arena tail **+ 128 MiB** |
| **−5536 MiB** | **1** | 0 | arena tail + 128 + **1258 MiB** ← the excursion |

The **128 MiB** quantum is identified in code. `weights_arena.py`:

* `arena_refill()` verifies the copy with `uint8_checksum(dst)` on device.
* `uint8_checksum` splits the ~4 GiB payload and sums each chunk with
  `chunk.sum(dtype=torch.int64)`, which casts first — so the transient is
  `8 × chunk`.
* `_checksum_chunk_bytes()` sizes the chunk adaptively:
  `min(16 MiB, free × 0.25 / 8)`, floored at 1 MiB.
* At 16 MiB the transient is `8 × 16 = 128 MiB` exactly. When it is served from
  torch's cache instead of the driver, NVML free never moves — hence the
  −4150 variant.

So the two modal values are the same allocation with and without a
driver-visible checksum buffer. **This is not a defect**; it is priced,
bounded, and documented. It is recorded here because it establishes that the
step is a *deterministic sum of identifiable terms*, which is what makes the
third value an anomaly rather than a tail.

---

## 3. The excursion: everything else was identical

`decompose` output, verbatim, target against the modal flip of the same boot:

```
== decompose rank 0 tp_to_pp
   TARGET 11:44:06: transient 7055 MiB, baseline 7723, trough 668
   MODAL  11:31:34: transient 5800 MiB, baseline 7725, trough 1925

   stage                         target     modal     delta
   plan                               0         0         0
   kv_pack                         -106      -106         0
   kv_local_read                    -76       -76         0
   allocator_cache_release          358       356         2
   backing_restore_span           -5280     -5280         0
   backing_release_span            3584      3584         0
   backing_restore                    0         0         0
   kv_write                           0         0         0
   gdn_state                          0         0         0
   weights_refill                 -5536     -4278     -1258   <== THE DIFFERENCE
   cutover                          456       456         0
   done                               0         0         0
```

Corroborating invariants at the anomalous flip (epoch 134):

* **Entry state byte-identical** to the modal flip: `free 6203 MiB, slack 524`
  immediately before `weights_refill` in both.
* **Exit slack identical**: 652 MiB in both.
* **Seam traffic normal**: `staging reserved 4148.35 MiB` — the same constant on
  all 86 flips; `sent 78.63 / received 169.26 / local 131.05 MiB`, all inside
  the ordinary spread.
* **No branch difference.** A mechanical diff of the whole flip block
  (arm → DONE, numbers normalised) shows exactly one added line: the census's
  own `CORRIDOR LAW BROKEN` marker. No alternate code path was taken.
* **The dip is real, not an instrument artifact.** The independent 100 ms NVML
  corridor sampler recorded `gpu1_free = 669` at `11:44:06.436`, against the
  census's 668 — the two instruments agree to 1 MiB, as they did in window 1.
* **Duration ≈ 280 ms**, three consecutive 140 ms samples
  (`2035 → 1405 → 1137 → 669`), recovering to 1287 within one sample.

---

## 4. What the 1258 MiB is — ranked suspects, each with its discriminator

Not resolved from the artifacts in hand. **This is the honest state**, and the
discriminator for each suspect is cheap. Ranked by fit to the evidence.

At the breach the runtime printed:

```
CORRIDOR LAW BROKEN during tp_to_pp rank 0 at stage 'weights_refill':
free 668 MiB is below the 1024 MiB floor. torch is holding 652 MiB of slack
(reserved 32342 MiB, allocated 31689 MiB)
```

> **`reserved` is not a physical quantity on this rank, and that is settled
> from the artifacts, not inferred.** The card's NVML total is **32088.5 MiB**
> (`s51/boot_arm_280000.log`: `nvml_used=30050.1 nvml_free=2038.4
> nvml_total=32088.5`; the two 3080s read `nvml_total=20054.9`). So at the
> breach `reserved 32342 + free 668 = 33010` overshoots the card by **921
> MiB** — and the same line shows the overshoot is not a crash artifact: **at
> rest**, `reserved=32040.0` against `nvml_used=30050.1` is already **1990 MiB
> of reserve that no physical page backs**. Expandable-segments virtual
> reservation is the ordinary explanation.
>
> **Consequence, and it corrects the obvious instrument fix.** Printing
> `reserved` per stage would *not* decide anything: a virtual reserve can grow
> without costing a byte of NVML free. **`allocated` is the decisive column** —
> live torch bytes, physically backed by definition. See §7.

### S1 — caching-allocator segment growth during the checksum leg (LEADING)

`uint8_checksum` requests one `8 × chunk` buffer per chunk over ~260 chunks. If
segment reuse fails for a run of chunks (size-class mismatch after the seam's
160 `backing_restore_span` / 256 `backing_release_span` operations churned the
arena), torch grows its reserve from the driver rather than reusing cache.
`slack` is unchanged at exit because the growth is *reserved and allocated*
together.

**Fits:** the post-event regime. After 11:44:06 the rank-0 baseline sat
**~740 MiB lower** (7725 → 6986/7074/7234) for **three minutes** and recovered
to exactly 7725 at 11:47:39 — a reserve that grew and was returned gradually.
The corridor guard reclaimed 310 MiB "from [allocator-cache]" 0.5 s after the
breach, i.e. there *was* fresh reclaimable cache that a modal flip does not
leave behind.

**Discriminator:** **`allocated`** across `weights_refill`. Under S1 the
8x-chunk buffers are live tensors while they sum, so `allocated` rises with
`free` falling. The census **already samples `allocated` at every mark and
prints only `reserved - allocated`** (`phase_flip_seam_census.py:277-280`);
printing `allocated` as its own field is a one-line change and needs no new
probe. Note S1 predicts a rise far larger than the 128 MiB the checksum is
capped at, so a ~1258 MiB rise in `allocated` indicts the *allocator's
behaviour around* the checksum rather than the checksum's own bound.

### S2 — a concurrent allocation by another actor on device 0

The census charges a stage the *whole* free-memory delta across it, so anything
else allocating on device 0 during the ~1 s stage is attributed to
`weights_refill`. Rank 0 was quiescent at the cutover, which makes this
unlikely but not excluded.

**Discriminator:** the same `allocated` column, read together with *who* was
running. S2 also raises `allocated` (any torch actor does), so it is not
separated from S1 by the number alone — it is separated by the concurrent log
context on device 0 in that second, which for this flip shows the rank
quiescent at the cutover. Ranking S1 above S2 rests on that, and on S1's
independent fit to the three-minute post-event baseline shift.

### S3 — VMM commit fragmentation in `arena_carrier.set_active_prefix`

`_commit_refill_high_water()` re-commits ~4150 MiB of physical pages behind a
stable VA. If the driver could not satisfy the commit from the handles released
on the previous `pp_to_tp` leg and had to back it with fresh pages while the old
ones were still held, free drops by more than the tail.

**Discriminator:** **`allocated` flat while `free` drops ~1258 MiB** — the
commit is outside torch entirely, so no torch counter moves. This is the one
clean, unambiguous split in the set: it separates S3 from {S1, S2} on a single
column. Secondary: instrument `set_active_prefix` to log the pages it actually
created versus reused.

### S4 — the `restore=` arm of `arena_refill` firing

A checksum mismatch rewrites the other layout. **Excluded by the artifacts:**
that path raises `WeightsArenaError` unconditionally after restoring, and the
flip completed with a normal `DONE` line and no error. Recorded so it is not
re-proposed.

### S5 — a bigger adaptive checksum chunk

**Excluded by arithmetic:** `_checksum_chunk_bytes` is capped at 16 MiB, so the
checksum transient cannot exceed 128 MiB by construction. 1258 ≫ 128.

### Families explicitly checked and NOT implicated

| Family | Why not |
|---|---|
| deep-prefill allocator transient (#493) | measured under a *batch*; the seam runs between batches, and `kv_pack`/`kv_local_read`/`kv_write` are identical to the MiB across the anomalous and modal flips |
| seam staging | `staging reserved 4148.35 MiB` on all 86 flips, including the anomalous one |
| graph capture | boot-time only; no capture line anywhere near 11:44 |
| radix eviction burst | pool census identical before and after the cutover: `size=280000 free=264624 cached=15376 unaccounted=0` |
| EAGLE 2×-KV reserve (#486) | the PP phase has **no** drafter; rung 2 *spilled* 456 MiB of draft weights at this flip exactly as at every other, and the `cutover +456` step is identical |

---

## 5. The modal draw is the larger finding, and it closes exactly

Independently of the outlier, the *routine* seam is under-funded, and the
arithmetic closes to the MiB:

```
  entry (TP phase at rest)                      7725 MiB free
  allocator_cache_release                       +356          -> 8081
  kv_pack + kv_local_read                       -182
  backing_restore_span / backing_release_span   -5280 / +3584
  --------------------------------------------------------------
  pre-refill level                                             -> 6203
  weights_refill  = arena tail 4150 + checksum 128   -4278     ->  1925   <- the corridor minimum
  cutover (draft weights spilled)                    +456      ->  2381
```

So `transient = wave_residual (1522) + arena_tail (4150) + checksum (128) = 5800`.

The same reconciliation on s51's modal flip (12:41:59) closes identically at a
different left term: `7314 − 1680 − 4150 − 128 = 1356`. **The whole boot-to-boot
difference in the modal transient — 5958 against 5800 — is in the wave residual
(1680 against 1522); the refill step is `−4278` in both.** So even the routine
boot-to-boot movement of the corridor minimum is a movement of the *unfunded*
term, not noise in the funded one.

**What the gate funds.** `PhaseFlipRuntime._staging_bytes()` returns

```python
max(wave_peak, self._draft_restore_bytes(direction), self._arena_tail_bytes(direction))
```

— `max()`, not `sum()`. The justification in the comment is that *"the two peaks
belong to different legs and cannot coexist"*, which is true of the pp→tp
drafter restore versus the tp→pp arena tail. **It is not true of the wave
residual versus the arena tail on the same leg**: the census shows the backing
restore is a *persistent step into the PP layout* that is still held when the
refill begins, 1522 MiB below entry. The estimate is therefore
`staging reserved 4148.35 MiB` — logged identically on all 86 flips — against a
measured draw of 5800.

| | MiB |
|---|---:|
| runtime seam-entry estimate (`staging reserved`) | 4148 |
| measured modal draw | **5800** |
| estimator short by | **1652** |
| measured worst draw | **7055** |
| estimator short by | **2907** |
| planner `seam_staging_mib` (pp_cut.py:203, default) | **0.0** |

**Two gates, two distinct defects.**

1. **Runtime seam-entry gate** (`phase_flip_runtime.py:4341+`, C20). Armed and
   working — it refuses and delays on the `pp_to_tp` leg (171 guard events in
   s50, 4 delays). But its estimator composes the tp→pp leg with `max()` and so
   predicts a trough of `7725 − 4148 − 512 = 3065` where the real trough is
   1925. It clears trivially and never logs, which is why the leg *looks*
   ungated in the log. **It is gated; the model is wrong on this leg.**
   Note the guard's own comment warns against exactly the inference that a
   quiet log means an inert gate — the log is quiet because the ask is
   satisfied from free, not because the check is absent.

2. **Planner cut gate** (`planner/pp_cut.py`). `seam_staging_mib` has **no
   producer anywhere in the tree** (`grep` outside `pp_cut.py` returns nothing)
   and defaults to `0.0`, so `runnable_headroom_mib == headroom_mib`. The
   field's own docstring says it: *"left at zero the verdict is a residency
   verdict again and says nothing about runnability."* The `40,12,12` admit
   with "374.9 MiB headroom" is therefore a **residency** verdict that funded
   **0 MiB** of a 5800 MiB seam.

3. **The #485 transient census cannot see the seam.** `transient_census.note()`
   is called from exactly one site — `scheduler.py:6423`, inside
   `process_batch_result`, labelled `batch.forward_mode.name`. The cutover is
   not a batch, so **no seam sample is ever taken**. The census's measured
   worst (`1989 MiB` planner cut / `3148 MiB` ship, per its own docstring) omits
   the largest transient in the system by a factor of three. BLOCKER 2's fix
   made the gate honest about *load-state* transients; the *seam* transient was
   never in its reference class.

**Calibrated value, supplied.** The runsheet's known limit
*"`seam_staging_mib` remains uncalibrated"* can now be closed for this rank and
cut, from 196 measured flips:

```
seam_staging_mib(rank 0, tp_to_pp, cut 40,12,12 / attn 10,3,3, pool 280000)
    = 5800 MiB modal, 7055 MiB worst observed
```

---

## 6. Do the two HIGH threats remove the excursion source? — **No.**

`RUNSHEET` §3 flagged two commits as HIGH and asked whether a re-baseline makes
s50's class obsolete. Checked by path, on `2b71b5b242`:

| Commit | Touches | Verdict |
|---|---|---|
| `58660f2d7f` *"Host shmem is a priced axis"* | `weights_arena.py` **+137**, `host_shmem.py` (new), `memtier/profile.py`, `scheduler.py` | **Does not touch the excursion source.** The only change inside `weights_arena.py` is the **host** image allocator: `torch.zeros(pin_memory=True)` (power-of-two rounded) → exact-size `MAP_SHARED` + `cudaHostRegister`. `arena_refill`, `uint8_checksum`, `_checksum_chunk_bytes`, `_CHECKSUM_FREE_SHARE`, `set_active_prefix` and `_commit_refill_high_water` are **untouched** — verified by grepping the diff for each symbol. It removes 13.65 GiB of *host* rounding, which is BLOCKER 1's mechanism, not this one. |
| `cb8da83774` *"pay the normalisation on the host, not the corridor"* | `kv_backing_relief.py` only (+51) | **Does not touch the excursion source.** It moves cost off the `backing_restore_span` / `backing_release_span` stages — which net to `−5280 / +3584` *identically* in the anomalous and modal flips. It cannot change `weights_refill`. |

**Stronger statement, and it settles the question.** Across the entire merge
window `0ae49fafb4..2b71b5b242` (the whole span since HANDOFF 695), a
`git log -p` over `weights_arena.py` and `phase_flip_boot.py` filtered to
`_CHECKSUM_FREE_SHARE`, `_CHECKSUM_CHUNK`, `_checksum_chunk_bytes`,
`uint8_checksum`, `arena_refill`, `def refill`, `set_active_prefix`,
`_commit_refill_high_water` returns **nothing**. The device-side refill
primitives are unchanged.

> **Therefore the argument "re-baseline makes s50's class obsolete" is NOT
> available.** The excursion source is present on the current line, unmodified.
> This must not be attempted in the flip proposal.

**What the re-baseline *is* still worth.** `cb8da83774` moves normalisation cost
off the corridor, which should *raise* the at-rest TP free on rank 0 — the LEFT
term of the identity in §0, and the only term that a code change has moved. That
is directly the quantity §7's criterion needs. So B0/B1/B2 remain worth running,
but for the left term, not as certification.

---

## 7. AMENDED CERTIFICATION CRITERION

**C2 as written is the wrong statistic.** `margin > spread` computes the spread
over *window minima* — two numbers — while each window contains ~100 flips. A
window's minimum is already an extreme-value statistic over its own flips;
comparing two of them discards 194 samples to keep 2. Worse, it measures the
wrong population: the next flip draws from the *transient* distribution, not
from the distribution of window minima.

**C2′ (replaces C2).** For the binding rank, over every flip in the reference
class:

```
   at_rest_free(binding rank, TP phase)  -  max_observed_transient  >=  1024 MiB
```

Applied to the real data, verbatim from the tool:

```
== C2' amended: margin against the WORST observed seam transient
   reference class: 196 flips over 2 window(s)
   worst transient: 7055 MiB (at 11:44:06, stage weights_refill)
   required at-rest free: 7055 + 1024 = 8079 MiB

   FAIL  s50   modal baseline 7725  observed min  668  margin vs worst  -354 MiB
   FAIL  s51   modal baseline 7314  observed min 1354  margin vs worst  -765 MiB

NOT CERTIFIED (C2')
```

**C4 (new, and it is the one that matters going forward).** A cut may not be
called certified while the gate that admitted it funds `0.0` MiB of seam
staging. Either

* `seam_staging_mib` is supplied from measurement (§5 gives the value), and the
  planner gate re-run with it — at which point `40,12,12`'s 374.9 MiB of
  headroom against a 5800 MiB seam is refused by the gate itself, without a
  window; **or**
* the seam draw is *reduced* below the available headroom, and the reduction is
  measured.

**C5 (new).** The `transient_census` must take at least one sample inside the
cutover, or its `worst_transient_mib` may not be quoted as "the worst state the
deployment will serve". Today it structurally cannot see the largest one.

**C3 is retained unchanged** (the binding rank must be the same card in every
window). It holds: rank 0 in both windows, and ranks 1–2 measure a transient of
exactly 0.

### What one boot must show

Not "no breach". A boot that does not breach has only shown it did not draw the
outlier — s51 is the worked example, and it had the *thinner* margin. One boot
must produce **three specific readings**:

1. **`allocated` per stage mark**, not just `slack`. `allocated` flat across
   `weights_refill` while `free` drops says S3 (a VMM commit outside torch);
   `allocated` rising with `free` says S1 or S2. That is the one clean split,
   and it is a one-line change to a log format on a path that already samples
   the value. Print `reserved` too, but read it knowing it is virtually
   inflated by ~1990 MiB at rest on this rank (§4) — it cannot carry this
   decision on its own.
2. **The at-rest TP free on rank 0 on current HEAD** — the left term of the §0
   identity, and the only term `cb8da83774` could have moved. Needed: ≥ 8079 MiB
   to clear C2′ against the worst already observed. s50 had 7725, s51 had 7314.
3. **The transient distribution over that window's ~100 flips**, pooled with the
   existing 196. Every additional window narrows the right tail; that is what
   windows are *for* under C2′, and it is a use of a window that actually
   accumulates.

If reading 2 comes back below 8079 MiB — as both prior boots did — the
configuration is refused by arithmetic and no further window is needed.

---

## 8. READY METAL TICKET — for the GPU shift

**Title:** #485 — instrument the seam's reserved column and re-baseline the
binding rank's at-rest free on HEAD.

**Not a certification window.** Do not book this as window 2 or 3.

**Precondition (desk, ~10 min, may be done by the GPU shift or handed back):**
one-line change in `python/sglang/srt/managers/phase_flip_seam_census.py`
around line 277 — the formatter already holds `free`, `reserved` and
`allocated` and prints `free`, `step` and `slack = reserved - allocated`. Print
`reserved` and `allocated` as their own fields. No behaviour change, no new
probe, no new cost. Without it, boot 1 below answers nothing that the two
existing windows have not already answered.

**Boots, in order:**

| # | Config | What it must record |
|---|---|---|
| **M0** | ship `14,10,8`, pool 620000, flip ON | at-rest TP free per rank on HEAD; MemAvailable min; `oom_kill` delta. Separates `cb8da83774`'s code move from the cut. |
| **M1** | arm `40,12,12` / attn `10,3,3`, pool 280000, flip ON, `SGLANG_UNEVEN_TOKEN_VECTOR=10,3,3` | the three readings of §7. ≥ 22 min so the flip count is comparable to the existing 196. |

Common knobs unchanged from the runsheet §5.0: `--rank-gpu-memory-mib
31400,19300,19300`; corridor sampled at 100 ms on the NVML FREE column;
seam census left on (it is on by default).

**Answer these four questions, in writing, from M1's log:**

1. **Confirm the virtual reserve.** `reserved` overshoots this card's
   32088.5 MiB total by 921 MiB at the breach and by 1990 MiB *at rest*. Check
   `PYTORCH_CUDA_ALLOC_CONF` / `expandable_segments` in the boot env and record
   it, so the next reader does not repeat the desk's first mistake of treating
   `reserved` as physical. This is bookkeeping, not a blocker.
2. **S3 vs {S1, S2} — the decision this boot exists for.** Across
   `weights_refill`, does `allocated` stay flat while `free` drops (S3: a VMM
   commit outside torch), or does it rise with `free` (S1/S2: a torch
   allocation)? One window decides it, on one column.
3. **The left term.** What is the at-rest TP-phase free on rank 0 on HEAD, for
   both M0 and M1? Is it ≥ 8079 MiB? State the number even if the answer is
   obviously no — it is the input to option (c).
4. **Does the outlier recur?** Pool M1's `tp_to_pp` rank-0 transients with the
   existing 196 using `excursion_485.py census`. Report the new worst and the
   new n. A second 7055-class event changes this from "one anomalous flip" to
   "a rate", and that is a different problem with a different fix.

**Do NOT do in this ticket:** propose the default flip; run the A/B depth probe
(the +25.5 % gain is settled and §0 of the runsheet forbids re-litigating it);
or treat a clean corridor as certification.

**Teardown:** unchanged from `RUNSHEET_485_CERTIFICATION.md` §8. Stop the
heartbeat before releasing `/spinning/gpu-arb/holder`. Whoever stopped serving
owns bringing it back.

**Follow-on tickets this analysis creates (desk, no metal):**

* **T1** — supply `seam_staging_mib` from the seam census and re-run the planner
  gate on `40,12,12`. Expected: the gate refuses its own prior admit. That is a
  CPU-only falsification of the admit, and it does not need a window.
* **T2** — `_staging_bytes` composes the tp→pp leg with `max()`; the wave
  residual and the arena tail coexist on that leg. Red-first test: a runtime
  whose wave residual is non-zero and whose arena tail is the max must estimate
  their **sum**, not the max. Fix behind the existing estimator, no new gate.
* **T3** — `transient_census` takes no sample inside the cutover
  (`scheduler.py:6423` is the only `note()` site). Add a seam load state so the
  #485 census and the seam census stop disagreeing by a factor of three.

---

## 9. Desk validation performed for this document

| Item | Result |
|---|---|
| `excursion_485.py smoke` | **8/8**, red-on-demand: parser on a noisy log, transient derivation, stage-step recovery, absent stage, judge REFUSES a clean window whose baseline cannot absorb the class worst, judge CERTIFIES when it can, decompose marks exactly one stage, and it is the right one |
| `excursion_485.py census` on real s50 + s51 | 196 flips, distributions as tabulated in §1 and §2 |
| `excursion_485.py decompose` on real s50 @ 11:44:06 | reproduces §3 verbatim; one stage differs |
| `excursion_485.py judge` on real s50 + s51 | `NOT CERTIFIED (C2')`, s50 −354 MiB, s51 −765 MiB |
| Threat-commit path check | §6, by grepping each diff for the named symbols and by `git log -p` over the whole merge window |
| `ruff check scripts/cert_485/` | clean |

No GPU, no serving process, no model touched.
