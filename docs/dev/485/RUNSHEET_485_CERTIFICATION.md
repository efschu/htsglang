# RUNSHEET — #485 planner-cut certification, windows 2 and 3

**Purpose.** Close the open gate on `--pp-solve-cut`: *"2nd + 3rd clean
certification windows"* before a default-flip proposal goes to the user. Two
windows, one runsheet, criteria pre-registered in code so they cannot be
adjusted afterwards to fit the result.

**Derived from** `integration/r2` @ `cb8da83774` (re-resolved 2026-08-14; the
line had advanced from `9cedf43811`). Branch `chore/ticket-485-cert`,
worktree `/spinning/wt-485-cert`, `PYTHONPATH=/spinning/wt-485-cert/python`.

> ### REVISION 2 — 2026-08-14, desk shift, branch `feat/desk-485-excursion`
>
> **This copy supersedes the one in `/spinning/wt-485-cert`, which is left
> untouched.** Re-derived from `integration/r2` @ `a0db672d5f` (MERGE-R11).
>
> §2 chose path **(b)** — *explain the excursion*. It has been executed, at
> the desk, from artifacts already on disk. The result changes this runsheet
> in four places, and the first one changes what the windows are FOR:
>
> 1. **The excursion is a stage, not a variance** (§1 revised). It is named,
>    quantified over 196 flips, and it is not boot-to-boot noise.
> 2. **The planner gate now refuses `40,12,12`** once the measured seam is fed
>    to it. **The windows as previously written would boot a cut the gate no
>    longer admits** — see §5, which is rewritten around that.
> 3. **C2 is replaced by C2′, and C4/C5 are added** (§6). Under C2′ both
>    existing windows FAIL.
> 4. **The +25.5 % number is conditionally retired** (§6a).
>
> Full derivation: `docs/dev/485/EXCURSION_ANALYSIS_485.md` on the same branch.
> Reproduce any number in it with `scripts/cert_485/excursion_485.py`.

---

## 0. What is NOT in doubt, so the window does not spend time on it

**The gain is settled.** Window 1 measured a same-shift, same-code ship control:

| arm | pool | median @ 179200 | spread | n scored |
|---|---:|---:|---:|---:|
| ship control `14,10,8` | 620000 | **81.878 s** | 0.308 % | 5 |
| cut `40,12,12` / attn `10,3,3` | 280000 | **65.257 s** | 1.666 % | 5 |

`81.878 / 65.257` = **+25.5 % throughput**, equivalently **20.3 % less wall
time**. A 25 % effect against a 0.3–1.7 % instrument spread is roughly 15× its
own noise, and the arm reproduced the previous shift to **0.08 %**. **No
further window is needed to believe the speed.** Windows 2 and 3 are not about
throughput and must not be justified by it.

> Do not quote the older **+50.9 %**. That figure divides by successor 48's
> control, which differs in KV pool *and* four shifts of code and cannot be
> decomposed. The number that belongs in a recommendation is the one whose
> denominator is what actually boots: **+25.5 %**.

> **CONDITIONAL, as of 2026-08-14 — read §6a before quoting this.** The
> +25.5 % is settled *for the cut `40,12,12`*. It is a property of the CUT, not
> of the feature. The planner gate, once fed the measured seam staging, may no
> longer admit `40,12,12` (§5.1b, phase P2) — and if it admits a different cut,
> **this number is retired, not carried over**. "The gain is settled" remains
> true; "the gain is settled for whatever we end up certifying" was never
> established and must not be assumed.

---

## 1. What IS in doubt — ~~and it is a variance, not a mean~~
### (title superseded: it is not a variance either — see the revision at the end of this section)

The same configuration — `40,12,12 / attn 10,3,3`, pool 280000, flip ON — was
booted twice. Computed here directly from the raw NVML series, not quoted:

| boot | gpu0 min | **gpu1 min** | gpu2 min | NVML breaches | seam breaches | ranks alive |
|---|---:|---:|---:|---:|---:|---:|
| **s50** (`corridor_planner.csv`) | 5585 | **669** | 6075 | **2** | **2** | **2 of 3** |
| **s51** (`corridor_arm.csv`) | 5591 | **1355** | 6125 | 0 | 0 | 3 of 3 |

**The variance is rank-specific and that is the finding.** gpu0 reproduces to
**6 MiB** and gpu2 to **50 MiB**. gpu1, the binding rank, moves **686 MiB** —
against a measured margin above the 1024 MiB floor of **331 MiB**.

> `HANDOFF_695.md` §5.1: *"~690 MiB of boot-to-boot spread on the binding rank
> against a 331 MiB margin — the margin is inside the spread."*

Both instruments agree: seam census deepest trough 668 MiB (s50) against
1354 MiB (s51), the latter within **1 MiB** of NVML's 1355.

**So the certification question is: does the binding rank's minimum hold across
boots, or was s50 the normal case and s51 the lucky one?** One more clean
window cannot answer that. Two can bound it, weakly — see §6 on what 3 windows
actually support.

### REVISED 2026-08-14 — the question above is the wrong question

The minimum is not a draw from a distribution. It is an identity, and every
term is measured:

```
  min_free(rank 0) = at_rest_TP_free - wave_residual - arena_tail - checksum
   s50 modal:  1925 = 7725 - 1522 - 4150 - 128        (seam transient 5800)
   s51 modal:  1356 = 7314 - 1680 - 4150 - 128        (seam transient 5958)
   s50 breach:  668 = 7723 - 1522 - 4150 - 128 - 1258 (seam transient 7055)
```

Pooled over **196 `tp_to_pp` flips** across BOTH windows, on rank 0: the
transient body is 5654–6008 MiB and **one** flip sits at 7055. The step
charged to `weights_refill` takes exactly three values in 196 flips — −4150
(the arena tail), −4278 (tail + the 128 MiB adaptive-checksum transient), and
once −5536. **Ranks 1 and 2 report a transient of 0 on all 86 flips**, which is
why gpu0 and gpu2 reproduce to 6 and 50 MiB: only the binding rank carries a
seam transient at all.

**And s51 was not the lucky boot.** It ran with the *thinner* routine margin —
332 MiB against s50's 901 — and simply never drew the outlier. The same seam
during s51 would have troughed at **259 MiB**.

So "does the minimum hold across boots" mis-frames it. The minimum holds
exactly as long as the configuration is not asked to absorb a transient nobody
budgeted. Which brings us to the finding that governs this whole document:
**no gate funds that transient.** `RankResources.seam_staging_mib` defaulted to
`0.0` and **nothing in the tree supplied it**, so the `40,12,12` admit "with
374.9 MiB headroom" was a *residency* verdict — the field's own docstring says
so — that funded 0 MiB of a 5800 MiB seam.

---

## 2. BLOCKING FINDING — is s50 still in the reference class?

**This must be answered BEFORE the windows, because it decides what they mean.**

`HANDOFF_695` closed two blockers after s50:

* **BLOCKER 1** — s50's rank death was a **host** SIGKILL (cgroup OOM), not a
  VRAM event. Container ceiling ~120 GiB, real working margin ~45 GiB;
  schedulers held **75.1 GiB** of shmem; `oom_kill` had fired **9** times.
  Landed: signal-naming and cgroup-OOM attribution in `utils/watchdog.py`.
* **BLOCKER 2** — the wired `--pp-solve-cut` path never set `transient_mib`, so
  it took its `0.0` default and admitted `42,11,11`, which metal measured
  breaching. Landed: `planner/transient_census.py`; the gate now charges the
  WORST load state and **REFUSES** a census with no measured transient.

**The unresolved question.** BLOCKER 1 explains the *rank death*. It does not,
on the evidence in hand, explain the **gpu1 corridor breach to 669 MiB** — a
VRAM quantity. BLOCKER 2's fix changes which cut is *admitted*, but s50's
window already ran `40,12,12`, which is what the fixed gate also picks.

Two readings, and the window's value depends entirely on which is true:

| Reading | Consequence for certification |
|---|---|
| **A — s50's breach was downstream of the host kill** (dying rank perturbs the corridor) | s50 leaves the reference class. Certification restarts from n=1 (s51), and windows 2+3 give 3 clean boots. |
| **B — the breach is independent boot-to-boot variance** | The reference class is 1 breach in 2 boots. Two more clean windows give **1 in 4 = 25 %** observed breach rate, which does **not** support a default flip. |

### RESOLVED AT THE DESK, 2026-08-14 — and the answer is reading B

The corridor CSV carries `ts_ms`, so the ordering is answerable without a
window. Run directly against
`/spinning/evidence-631/s50/corridor_planner.csv` and
`boot_gate_cut_280000.log`:

| event | time | source |
|---|---|---|
| corridor series starts | 11:30:53Z | `corridor_planner.csv` |
| **gpu1 crosses the floor — min 669 MiB** | **11:44:06Z** | `corridor_planner.csv`, the only excursion |
| all three ranks decoding normally | 11:48:41 | log:49381-49383, `PP0`/`PP1`/`PP2` all generating |
| `scheduler_0 crashed with exit code -9` | **11:48:55** | log:49494 |
| corridor series ends | 11:54:53Z | `corridor_planner.csv` |

**The breach preceded the SIGKILL by 4 minutes 49 seconds, and the server was
fully healthy throughout that interval** — all three ranks were still producing
decode batches 14 seconds before the kill. At the breach sample gpu0 read 9257
and gpu2 6857 MiB, so this was a *localised* excursion on the binding rank, not
a global collapse.

**Therefore reading B holds: s50's corridor breach is NOT an artifact of a
dying rank, and s50 STAYS in the reference class.** BLOCKER 1 explains the
death; it does not explain the breach, and the ordering rules out the reverse
direction.

### BLOCKING CONSEQUENCE — two more windows cannot close this gate

The reference class is **1 breach in 2 boots** of the identical configuration.
Two more clean windows make it **1 in 4 (25 %)**. A 25 % observed breach rate
on the corridor law does not support a default flip, and no arrangement of
three clean windows changes it, because s50 is *in* the sample.

**The gate as currently framed — "2nd + 3rd clean certification windows" — is
therefore not sufficient.** Running them is still worthwhile (they measure the
C2 spread, which is the real evidence), but the shift must know before it
starts that a CERTIFIED verdict is not reachable on this path alone. Three
options actually close it:

| Path | What it takes | Assessment |
|---|---|---|
| **(a) More windows** | Bound a low breach rate by repetition. With s50 in the sample you need enough boots to show the rate is small — many windows, not two. | Expensive, and it measures patience rather than the mechanism. |
| **(b) Explain and remove the 686 MiB excursion** | Find what makes gpu1 alone dip ~686 MiB between boots. gpu0 reproduces to 6 MiB and gpu2 to 50 MiB, so this is one rank's transient, not global noise — a strong hint it is the flip's staging on the binding rank. | **Preferred.** It converts a variance into a fixed term, and the gate already has a transient census that could carry it. |
| **(c) Buy margin until it exceeds the spread** | Choose a cut whose margin clears ~690 MiB rather than 331. HANDOFF 695 §3 records the **measured worst** admit as `36,15,13`, more conservative than `40,12,12`. Some of the +25.5 % is traded for a margin that survives the observed spread. | Concrete and available now. Requires re-measuring the gain at the safer cut. |

**Recommendation to the operator, not a decision:** run B0/B1/B2 as specified —
they cost one window and produce the C2 spread on current HEAD, which every
path above needs — but frame them as *characterising the excursion*, not as
certification. Then choose (b) or (c) with the spread in hand.

### PATH (b) WAS TAKEN AND IT IS DONE — 2026-08-14, no window spent

The excursion was explained from artifacts, and the explanation is §1 revised.
What it cost the certification, in order:

* **The mechanism is named.** One `tp_to_pp` seam whose `weights_refill` drew
  5536 MiB against the invariant 4278. Every other stage byte-identical to the
  MiB, identical entry state, no branch difference, and the independent 100 ms
  NVML sampler corroborates the trough to 1 MiB.
* **The residual unknown is bounded and has a named discriminator.** The
  1258 MiB excess itself is NOT resolved. Three suspects remain live and they
  are separated by ONE column — `allocated`, which the census now prints. That
  is the only thing left that needs metal.
* **The threats in §3 do NOT remove the source.** Both HIGH commits leave the
  device-side refill primitives untouched; a `git log -p` over the whole merge
  window filtered to those symbols returns nothing. **"Re-baseline makes s50's
  class obsolete" is not available and must not be argued.**
* **Option (c) is now measurable rather than a guess.** Buying margin means
  clearing `worst_transient + 1024` = **8079 MiB** of at-rest free on the
  binding rank. s50 had 7725; s51 had 7314.

**And the gate that admitted the cut has been fixed** (`_pp_cut_seam_staging`,
same branch). That is what forces §5 to be rewritten: this runsheet was
written to re-run `40,12,12`, and the honest gate may not admit it.

---

## 3. THREATS TO VALIDITY — what merged since window 1

**127 commits** landed between `0ae49fafb4` (HANDOFF 695) and `cb8da83774`.
Filtered to the binding mechanism by **path**, not by commit message:

| Component | Change | Threat |
|---|---|---|
| `planner/pp_cut_calibration.py` | **UNCHANGED** | **None.** Window 1's gate verdicts (REFUSES `42,11,11` over by 461 MiB naming load state `EXTEND`; ADMITS `40,12,12` with 374.9 MiB headroom) still describe this code. |
| `planner/transient_census.py` | **UNCHANGED** | **None.** The transient arithmetic is the one that was measured. |
| `planner/feasibility.py` | +19 lines: a `card_library` passthrough for the #413 buying advisor, documented as keeping `SEED_CARDS` so "every existing caller is unchanged" | **None** for this cut. |
| `mem_ledger/host_shmem.py` (**new, +419**) + `managers/scheduler.py`, `model_executor/weights_arena.py`, `memtier/profile.py` — `58660f2d7f` *"Host shmem is a priced axis: stop rounding the pinned flip images"* | **HIGH — re-baseline required.** Window 1 measured MemAvailable min **15.9 GiB** on the arm against 23.5 on ship, and host memory is the mechanism that killed s50. Changing how the pinned flip images are sized changes that number. **The 15.9 GiB figure is stale.** |
| `managers/kv_backing_relief.py` — `cb8da83774` (**HEAD**) *"pay the normalisation on the host, not the corridor"* | **HIGH — moves BOTH axes.** This explicitly relocates cost from the corridor to the host. It may **widen** the corridor margin (the certification's subject) and **narrow** the host margin (the thing that killed s50). Either direction invalidates a naive comparison against window 1's minima. |
| `mem_ledger/` others: `corridor_trace.py` (new, +291), `boot_history.py` (new, +351), `host_anon_644.py` (new, +257), `reconcile.py` (+630) | **MEDIUM.** New instrumentation around the same quantities. Confirm the corridor sampler still reports the FREE column identically before treating window-2 minima as comparable to window 1's. |

**Consequence, and it is not optional.** Windows 2 and 3 are **not** repeats of
window 1 under identical code. The GPU shift must:

1. **Re-baseline the host axis** — record MemAvailable min and `oom_kill` delta
   on this HEAD for both ship and arm, and state whether 15.9 GiB still holds.
2. **Re-baseline the corridor on the SHIP config first** (one window) so that a
   changed gpu1 minimum is attributable to the code move rather than to the cut.
3. Treat any corridor improvement as **suspect until explained** — HEAD moved
   cost off the corridor deliberately, so a wider margin may be the commit, not
   the cut.

---

## 4. Flags — verified against THIS tree by building the real parser

```bash
cd /spinning/wt-485-cert && export PYTHONPATH=/spinning/wt-485-cert/python
CUDA_VISIBLE_DEVICES=99 python scripts/cert_485/certify_485.py flags
```

**Verified 2026-08-14, all PASS on `cb8da83774`:** `--pp-solve-cut`,
`--pp-stage-ratio`, `--pp-attn-stage-ratio`, `--pp-layer-ratio`,
`--max-total-tokens`, `--rank-gpu-memory-mib`, `--rank-gpu-id`,
`--enable-phase-flip`. The `--phase-flip-*` family is exactly
`--phase-flip-policy`, `--phase-flip-purity`, `--phase-flip-spill-depth`,
`--phase-flip-tp-vector`.

Verified by **parsing, not grepping** — flags derived from annotated dataclass
field names are invisible to a literal search, which is how a grep-based check
reports a false MISSING (the #363 lesson).

---

## 5. The windows

### 5.0 Claim and common configuration

```bash
mkdir -p /spinning/gpu-arb && echo "485-cert $$ $(date -Is)" > /spinning/gpu-arb/holder
# heartbeat in its own process; STOP THE HEARTBEAT BEFORE RELEASING
```

Replay the captured ship argv/env with only the named flags changed
(`/spinning/evidence-631/s485/boot_arm.sh` is the pattern). Two knobs are held
COMMON across arms — experiment design, not drift:

* `--rank-gpu-memory-mib 31400,19300,19300` — uniform headroom so the budget is
  not itself a variable between arms.
* `SGLANG_UNEVEN_TOKEN_VECTOR` set **per arm to that arm's attention split**.
  The KV arena must follow the attention layers, or the arm measures a cut
  whose KV lives somewhere else.

### 5.1 Boots, in order

| # | Boot | Config | Purpose |
|---|---|---|---|
| **B0** | ship re-baseline | `14,10,8`, pool 620000, flip ON | §3 requires it: separates the code move from the cut. Record gpu minima + MemAvailable. |
| **B1** | **certification window 2** | `40,12,12` / attn `10,3,3`, pool 280000, flip ON, `SGLANG_UNEVEN_TOKEN_VECTOR=10,3,3` | The window. ≥ 22 min, corridor sampled at 100 ms. |
| **B2** | **certification window 3** | identical to B1, new output dir only | The repeat. **Nothing may differ**, or the pair stops being a repeat. |

---

### 5.1a SUPERSEDED — THE ORDER ABOVE IS NO LONGER RUNNABLE AS WRITTEN

**B1/B2 boot `40,12,12`. The planner gate, once fed the measured seam, may
refuse `40,12,12`.** Certifying a cut the admitting gate rejects is not a
certification; it is a manual override with a stopwatch. The table above is
kept so the change is legible, and replaced by §5.1b.

**The re-solve is a genuine fork, and the desk cannot call it.** The outcome is
decided by the per-rank `fixed_overhead_mib` that only the real census
supplies. Two desk proxies bracket opposite answers, which is exactly why this
must be re-solved on metal rather than predicted here:

| proxy | seam fed | solver result |
|---|---:|---|
| roomy fixture overheads `(0,0,0)` | 5800 | feasible, picks **`42,11,11`** — *the cut BLOCKER 2's fix REFUSED* |
| overheads calibrated to the real admit's 374.9 MiB headroom | 5800 | **no contiguous 3-stage cut of 64 layers fits at all** |

Both proxies are wrong, in opposite directions. Only the census decides.

**One structural result from the sweep IS transferable, and it is
counter-intuitive.** Under the tight proxy, shrinking the KV pool **never**
recovers feasibility — 280000 down to 40000, all refused — because the
shortfall sits in the FIXED terms (overhead + weights + corridor + seam), and
the pool is not one of them. The levers that did recover it were
`--rank-gpu-memory-mib` (**+6000 MiB per rank**) and reducing the seam itself
(down to ~500 MiB, and it then picks a *different* cut, `39,12,13`). So if the
re-solve returns nothing, **do not reach for `--max-total-tokens` first** — on
this shape it is the one knob that cannot help.

> Note for a follow-up ticket, not for this window: the seam-cap guard's own
> operator advice says *"lower `--max-total-tokens`, raise this rank's
> `--rank-gpu-memory-mib`, or choose a layer cut whose seam fits"*. The first
> of those three is the one that did not work in this sweep. Flagged as an
> observation, fixture-derived; not changed here.

---

### 5.1b THE ORDER THAT REPLACES IT — three phases, and the gate goes first

**The rule: measure the seam, feed the gate, let the gate choose the cut, THEN
certify what it chose.** Certification follows admission; it does not precede
it.

| Phase | Boot | Config | What it must produce |
|---|---|---|---|
| **P1** | **M0** ship re-baseline | `14,10,8`, pool 620000, flip ON | at-rest TP free per rank on HEAD; MemAvailable min; `oom_kill` delta. Separates `cb8da83774`'s code move from the cut. **Must actually FLIP**, or it writes no seam states and P2 has no input. |
| **P1** | **M1** arm characterisation | `40,12,12` / attn `10,3,3`, pool 280000, flip ON, `SGLANG_UNEVEN_TOKEN_VECTOR=10,3,3` | the four questions of §7a, including the `allocated` column that discriminates S1/S2/S3. **This is NOT a certification window** and must not be reported as one. |
| **P2** | *(no boot — CPU)* | re-solve with the fed gate | Run the planner against the census M0/M1 just wrote, with `seam_staging_mib` populated. **Record the cut it admits.** This is a desk step and costs no GPU time. |
| **P3** | **W1…Wn** | **the cut P2 admitted**, not `40,12,12` unless P2 picks it | the certification windows, judged under C2′. |

**P2 has three outcomes and all three are legitimate results:**

1. **It admits `40,12,12`.** The prior admit survives an honest gate. P3 runs
   as originally planned and the +25.5 % number stands (§6a).
2. **It admits a different cut.** *That* cut is what gets certified. The
   +25.5 % is **retired** and must be re-measured (§6a).
3. **It admits nothing at this pool/budget.** Then the configuration is
   refused by arithmetic, no window is owed, and the finding is the refusal.
   Report it as the result — it is a stronger outcome than a clean window,
   because it is not a sample. Consult the lever note in §5.1a before changing
   anything.

**Do not skip P2 to save a window.** P2 costs no GPU time and decides what P3
is even about. Running P3 first is how this ticket spent two windows measuring
a cut that its own gate would not admit.

> **WHICH CENSUS FEEDS P2 IS NOT THE SHIFT'S CHOICE — see §6b.** Two honest
> censuses gave opposite answers in the R12 window (M1 admits `37,14,13`, M0
> admits nothing). §6b pins the governing rule: **the worst value over the
> pooled population, per quantity**. A shift that solves against whichever
> census admits a cut is choosing its own result. And if the pooled-worst
> calibration admits nothing, **that is the answer**, reported with the lever
> arithmetic — not a window to be re-run against a friendlier input.

Each window: soak traffic for the duration, plus the A/B depth probe at 179200
with `n_scored >= 5` (window 1 scored 5; fewer is not a measurement).

**Sample validity — non-negotiable, and already enforced by the driver.**
Unique random prefix per sample, `/flush_cache` before each, and the driver
**rejects** any sample whose `meta_info` reports a cache hit over 5 %. Window 1
rejected zero. Note the known wrinkle: `/flush_cache` within ~400 s after soak
traffic on a PP instance reports `not-idle because: chunked_req, last_batch,
pp_microbatches` with 0 queued and 0 running — **one completed request clears
it**, so any cache-controlled benchmark straight after traffic fails its first
flush.

### 5.2 Corridor sampling — both instruments

NVML **FREE** column at 100 ms for the whole window; time-series minimum, never
a boot snapshot, never `total - used`.

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -lms 100 \
  > $OUT/corridor.csv &
```

The **seam census** is the second instrument and is not optional: on window 1
the two agreed to 1 MiB, so a disagreement is itself a finding.

Host axis, per §3:

```bash
bash scripts/hostmem_sample.sh > $OUT/hostmem.csv &   # MemAvailable + oom_kill
```

---

## 6. PASS / FAIL — pre-registered, and encoded in code

```bash
python scripts/cert_485/certify_485.py judge \
  --window w1=/spinning/evidence-631/s51/corridor_arm.csv \
  --window w2=$OUT_B1/corridor.csv \
  --window w3=$OUT_B2/corridor.csv \
  --arm    w2=$OUT_B1/ab_arm.json --arm w3=$OUT_B2/ab_arm.json \
  --seam-breaches w1=0 --seam-breaches w2=N --seam-breaches w3=N \
  --abandoned w1=0 --abandoned w2=N --abandoned w3=N \
  --ranks-alive w1=3 --ranks-alive w2=N --ranks-alive w3=N \
  --min-scored 5
```

**Per-window CLEAN** (W1–W7, all required): 0 NVML breaches below 1024 MiB on
every card; 0 seam-census breaches; flips in both directions with 0 ABANDONED;
soak `err == 0` and 0 tracebacks; 3 of 3 ranks alive; every scored sample
`cache_hit_frac <= 0.05` with 0 rejected; `n_scored >= 5`.

**Across windows:**

* **C1** every window CLEAN.
* ~~**C2** `margin > spread`~~ — **REPLACED by C2′ below, 2026-08-14.**
* **C3** the binding rank is the **same card** in every window. If it moves,
  the spread is computed over two different quantities and the windows are not
  repeats. *Retained unchanged; it holds — rank 0 in both windows, and ranks
  1–2 measure a transient of exactly 0.*

### THE GOVERNING CRITERIA, 2026-08-14

**C2′ replaces C2.** C2 computes a spread over *window minima* — two numbers —
while each window contains ~100 flips. A window's minimum is already an
extreme-value statistic over its own flips, so comparing two of them discards
194 samples to keep 2, and it measures the wrong population: the next flip
draws from the **transient** distribution, not from the distribution of window
minima.

```
   at_rest_free(binding rank, TP phase)  -  max_observed_transient  >=  1024 MiB
```

taken over **every flip in the reference class**, not over window minima.
**§6b defines what the reference class IS** — the pooled population, not the
cut's own flips — because that question was left open here and it decides the
verdict. Verbatim from `excursion_485.py judge` on the real data:

```
   reference class: 196 flips over 2 window(s)
   worst transient: 7055 MiB (at 11:44:06, stage weights_refill)
   required at-rest free: 7055 + 1024 = 8079 MiB

   FAIL  s50   modal baseline 7725  observed min  668  margin vs worst  -354 MiB
   FAIL  s51   modal baseline 7314  observed min 1354  margin vs worst  -765 MiB

NOT CERTIFIED (C2')
```

Note what C2′ does with windows: **every additional window narrows the right
tail** by adding ~100 flips to the reference class. That is a use of a window
that accumulates, unlike counting clean ones.

**C4 (new) — the gate must fund the seam.** A cut may not be called certified
while the gate that admitted it funds `0.0` MiB of seam staging. Either
`seam_staging_mib` is supplied from measurement and the planner re-run with it
(now enforced: `_pp_cut_seam_staging` **refuses** a census with no seam state
rather than defaulting), or the seam draw is reduced below the available
headroom and the reduction is measured. **C4 is what makes P2 mandatory.**

**C5 (new) — the transient census must be able to see the cutover.** Its
`worst_transient_mib` may not be quoted as "the worst state the deployment will
serve" while `note()` fires only in `process_batch_result`. Fixed on this
branch; C5 exists so a future reader can tell whether a given census predates
the fix — **a census with no `SEAM_*` load state is one that could not see the
seam, and C4's refusal will catch it.**

**Current verdict on the real data, reproduced by the tool:**

```
DIRTY  s50_arm   gpu0=5585 gpu1=669  gpu2=6075  binding=gpu1 margin=-355 MiB
CLEAN  s51_arm   gpu0=5591 gpu1=1355 gpu2=6125  binding=gpu1 margin=+331 MiB
NOT CERTIFIED (C1)
```

### What 3 clean windows actually support — state this, do not oversell it

**With reading B resolved (§2), s50 is in the sample, so the honest count after
windows 2 and 3 is 1 breach in 4 boots — a 25 % observed breach rate.** No
verdict of CERTIFIED is reachable on this path, and `certify_485.py judge` will
return NOT CERTIFIED (C1) for as long as s50 is included, which is correct.

Even setting s50 aside, counting clean windows is weak evidence: with zero
failures in three trials the rule of three puts the 95 % upper bound on the
breach rate at roughly **3/3 = 100 %** — three windows bound almost nothing.

What the windows *do* give is the **C2 statistic**: a measured spread of the
binding minimum across boots on current HEAD. That is the quantity the whole
gate turns on, and it is what paths (b) and (c) both need. **C2, not the count
of clean windows, is the evidence.** A proposal that says "3 clean windows" and
omits the spread is asserting the thing in question.

---

## 6a. THE +25.5 % NUMBER — what retires it, and when

**Added 2026-08-14, because §0 forbids re-litigating the gain and that
prohibition must not be smuggled across a change of cut.**

§0 is right that the speed is settled **for `40,12,12` against the ship control
`14,10,8` at pool 620000, same shift, same code**. It measured
`81.878 / 65.257 = +25.5 %`, arm spread 1.666 %, control spread 0.308 %, n=5.
That denominator is what makes the number quotable.

**It is a property of the CUT, not of the feature.** So:

| P2 outcome | status of +25.5 % |
|---|---|
| P2 admits `40,12,12` | **stands.** Same arm, same control, same pool. Quote it, and keep quoting the denominator with it. |
| P2 admits a **different** cut | **RETIRED.** It is not carried over, not scaled, and not cited as "approximately". The new cut moves layers between families and cards, which is precisely what the number measures. |
| P2 admits nothing | **moot.** There is no arm to have a gain. |

**What re-measuring costs, if it is needed.** The same A/B depth probe at
179200 that produced the original: `n_scored >= 5` per arm, unique random
prefix per sample, `/flush_cache` before each, driver rejecting any sample over
5 % cache hit. It must be run **against a ship control on the same boot and the
same code** — that is the whole reason the original number is trustworthy and
the older +50.9 % is not.

> **The failure mode this section exists to prevent.** Certifying cut X and
> then advertising cut Y's throughput. It would be easy to do accidentally: the
> gain lives in §0 under "what is NOT in doubt", the cut lives in §5, and
> nothing previously connected them. They are connected now.

Also note the ordering consequence: **the A/B probe belongs in P3, not P1.**
Running it during M1 measures the throughput of a cut that P2 may be about to
reject. §7a forbids it in the metal ticket for exactly this reason.

---

## 6b. WHICH CENSUS CALIBRATES THE GATE — the governing rule

**Added 2026-08-14, shift `m584`. This section exists because P2's answer
changed depending on an input nobody was reporting, and a gate with an
unreported free parameter is not a gate.**

`WINDOW_VERDICT_485_R12.md` §2 recorded the problem exactly and declined to
settle it: fed M1's census the gate **admits `37,14,13`**; fed the M0 ship-cut
census, at the same pool with the same budgets, the same code **admits
nothing**. "Two honest censuses give opposite answers. Nothing here says which
extrapolation is right." This section says which.

### THE RULE

> **A quantity that calibrates the gate is taken as the WORST value observed
> over the POOLED population of every census on the line — never the value from
> the census most convenient to the cut under test, and never the value from
> the cut's own class alone.**

Applied to the two places it bites:

| quantity | governing value | source |
|---|---|---|
| worst transient (C2′) | **7055 MiB** | pooled census, 221 flips (196 from s50+s51, 25 from M1); unchanged when W1's 99 are added — 320 observed |
| required at-rest free, binding rank | **7055 + 1024 = 8079 MiB** | the C2′ identity |
| seam staging, fixed overheads | the **most demanding** census on the line, per term | M0 vs M1, whichever refuses |

### WHY POOLED, when §6a argues the opposite

This is a real disagreement with a real argument on the other side, and it is
recorded rather than smoothed over. §6a and `WINDOW_VERDICT_485_R12.md` §3 hold
that a transient is **a property of the cut**: s50/s51/M1 measured `40,12,12`,
W1 measured `37,14,13`, and charging W1 with s50's 7055 pools across
configurations — the very error §6a exists to prevent, and the reason the
+25.5 % had to be retired. On W1's own class of 99 flips C2′ **passes by
+2589 MiB**.

That argument is correct about the **body** of the transient and wrong about
the **tail**, and the distinction is what decides it:

1. **The body is cut-specific and nobody disputes it.** W1's worst is 3186,
   M1's 4378, s50's modal 5800. These track the weights each stage refills, so
   they move with the cut, exactly as §6a says.
2. **The tail is not explained at all.** The 7055 is the invariant `−4278`
   step plus an excess of **1258 MiB** that `EXCURSION_ANALYSIS_485.md` states
   is **NOT resolved** — one flip in 320, every other stage byte-identical,
   identical entry state, no branch difference.
3. **The mechanism the tail rides on is cut-INDEPENDENT.** Q2 decided S3 over
   594 flips: the draw is the VMM commit in `arena_carrier.set_active_prefix`,
   with `max |Δallocated| = 0`. That is a property of the **flip machinery**,
   not of the layer cut. Q1 independently shows the arena is such a
   `cuMemAddressReserve`/`cuMemMap` reservation.

**An unexplained excess, riding on a mechanism shown to be cut-independent,
must be charged to every cut until it is explained.** Charging it only to the
cut that happened to draw it assumes precisely what is in question — that the
cut is why it was drawn. W1 did not draw a 7055; it also only ran once, and
s51 did not draw one either, then §1 showed s51 was not the lucky boot but the
one running the *thinner* margin, which would have troughed at 259 MiB had the
same seam arrived.

**So `37,14,13` is not certified by W1's own class.** Under the governing rule:

| window | at-rest free, binding | vs 8079 required | verdict |
|---|---:|---:|---|
| M0 ship `14,10,8` | 3953.7 | **−4125.3 MiB** | FAILS |
| M1 `40,12,12` | 2691.7 | **−5387.3 MiB** | FAILS |
| W1 `37,14,13` | 4435.7 | **−3643.3 MiB** | FAILS |

W1 passes C2′ on its own 99 flips by +2589 and fails the governing rule by
−3643. Both numbers are true; the second is the one that governs, and the
reason is stated above rather than asserted.

**This rule is falsifiable, and here is what falsifies it.** Explain the
1258 MiB excess. If its mechanism turns out to depend on the cut, the reference
class narrows to the cut's own class that same day and W1's +2589 becomes the
governing number. Until then the excess is unpriced, and an unpriced term
reading as free memory is the failure this whole ticket is about.

### THE CONSEQUENCE, STATED BEFORE THE WINDOW SO IT CANNOT BE RE-READ AFTER

**If the pooled-worst census admits nothing, that IS the certified answer.**
Not a failed window, not a measurement to be retried with a friendlier census
— the result. §5.1b outcome (3) already says a refusal "is a stronger outcome
than a clean window, because it is not a sample", and that applies here with
full force: a refusal derived from the worst observed population is a statement
about the configuration, while a clean window is one draw from a distribution
whose tail is known to contain a 7055.

**The named levers, and the one that is not a lever.** From §5.1a's sweep,
which is fixture-derived but structural:

* **`--rank-gpu-memory-mib`** — raise the binding rank's budget. This moves
  at-rest free, the left term of the C2′ identity, and is the lever that
  recovered feasibility in the sweep (+6000 MiB per rank there).
* **Seam reduction** — reduce the draw itself. The sweep recovered feasibility
  at a seam of ~500 MiB, and it then picks a *different* cut (`39,12,13`),
  so a seam reduction is not a free win on the same cut.
* **`--max-total-tokens` / KV-pool shrinking is NOT a lever.** Proven, not
  assumed: 280000 down to 40000, every one refused, because the shortfall sits
  in the FIXED terms — overhead, weights, corridor, seam — and the pool is not
  one of them. The seam-cap guard's own operator advice lists it first; on this
  shape it is the one knob that cannot help. **Do not reach for it.**

When the re-solve returns nothing, report the shortfall in the solver's own
numbers and convert it into both lever quantities — how many MiB per rank, and
how much seam reduction — so the refusal arrives with its remedy attached
rather than as a bare no.

### Housekeeping this rule implies

* The corridor sampler's long-form output is now read natively by
  `certify_485.py` (shift `m584`), so the ad-hoc `corridor_to_wide.py` in
  `evidence-631/m485/` is **superseded** and must not be reintroduced: it
  synthesised `ts_ms` from a file mtime, and `ordering` — which reasons about
  wall-clock instants — now refuses a series carrying no real timestamps.
* `certify_485.py judge` no longer prints `CERTIFIED` for C1+C2+C3 alone. C2′,
  C4 and C5 are attested with `--c2prime-margin`, `--cut` /
  `--c4-admitted-cut` and `--c5-seam-samples`; unattested is refused.

---

## 7. Default-flip proposal TEMPLATE

The flip itself is a **user decision**. This template is what goes to the user,
and it goes only when every "must hold" below is true. Fill the blanks with
measured values; a blank that cannot be filled is a reason not to send it.

> ### Proposal: make `--pp-solve-cut` the default? — DECISION REQUESTED
>
> **What it changes.** The planner solves the per-family layer cut instead of
> using the ship layout. Attention is compute-bound in deep prefill, so the
> solver puts both families more on the 5090 (`40,12,12`, attention `10,3,3`).
>
> **What it buys.** `+25.5 %` deep prefill at 179200 (`65.257 s` vs `81.878 s`),
> measured against a same-shift, same-code ship control. Arm spread `1.666 %`,
> control spread `0.308 %`, n=5 each.
>
> **What it costs.**
> * Host memory: `~8 GiB` more than ship (`15.9` vs `23.5` GiB MemAvailable min
>   — **re-measured on HEAD as ______**), in a container whose real working
>   margin is `~45` GiB and whose OOM killer has fired `9` times.
> * The gate prices **VRAM only** and still has **no host term**.
>
> **Corridor evidence.** Binding rank `______`, minima across `N` windows
> `______`, margin `______` MiB above the 1024 floor, observed boot-to-boot
> spread `______` MiB. Margin exceeds spread: `YES / NO`.
>
> **Reference-class statement.** s50's breach is `excluded / included` because
> `______` (the §2 time-ordering test).
>
> **Known limits, stated up front.**
> * Certified on one model, one depth (179200), one rig.
> * `seam_staging_mib` remains uncalibrated.
> * The transient table has **not** been tested across a cut boundary, so the
>   gate is shown *consistent*, not *predictive*.
>
> **Recommendation.** `______`. **The default flip is yours to make.**

**Gate on sending it at all — every one must hold:**

1. `certify_485.py judge` returns **CERTIFIED** (C1+C2+C3). **As of 2026-08-14
   this is not reachable by running windows 2 and 3**: §2 resolved that s50
   stays in the sample, so the path runs through option (b) or (c) of §2, not
   through window count alone.
   **REVISED:** the criteria are now C1 + **C2′** + C3 + **C4** + **C5** (§6),
   and C2′ is checked with `excursion_485.py judge`. C4 additionally requires
   that the cut being certified is the one an honest gate ADMITTED (§5.1b P2),
   not one certified in spite of it.
6. **The cut named in the proposal is the cut P2 admitted**, and the throughput
   number quoted beside it was measured on THAT cut (§6a). A proposal that
   certifies one cut and advertises another's speed is the specific error §7a
   exists to prevent.
2. §2's reference-class question is answered from the artifacts, in writing.
   **Done — reading B, breach 4m49s before the SIGKILL.** Carry that sentence
   into the proposal rather than re-deriving it.
3. §3's host axis re-baselined on HEAD; the `~8 GiB` delta restated or corrected.
4. The proposal names the +25.5 % denominator explicitly (never +50.9 %).
5. It states that the gate has no host term, since that is the mechanism that
   killed a window.

If 1–5 do not all hold, the honest output is a **status report**, not a
proposal. Sending a flip proposal whose margin sits inside its own spread would
be asking the user to ratify a coin flip.

---

## 7a. METAL TICKET — ONE WINDOW IF TIME PERMITS

**For the GPU shift, after #363. Everything below is executable as written; the
census format change it depends on is already merged on
`feat/desk-485-excursion`, so there is nothing to patch first.**

**Claim the window** per §5.0. Stop the heartbeat **before** releasing
`/spinning/gpu-arb/holder`.

### Order inside the window

| Step | Kind | Time | Notes |
|---|---|---|---|
| **M0** | boot | ~25 min | ship `14,10,8`, pool 620000, flip ON. **Must flip** — a boot that never reaches a cutover writes no `SEAM_*` state and P2 will refuse. |
| **M1** | boot | ~25 min | arm `40,12,12` / attn `10,3,3`, pool 280000, flip ON, `SGLANG_UNEVEN_TOKEN_VECTOR=10,3,3`. Characterisation, **not** a certification window. |
| **P2** | CPU | ~5 min | re-solve with the fed gate. No GPU. Records the admitted cut. |
| **W1** | boot | ~25 min | **only if P2 admits a cut** — certification window 1 on *that* cut, judged under C2′. If the window budget is gone, stop after P2: P2's verdict is the deliverable and W1 carries over. |

Census env for M0 and M1 — without these the boots produce no P2 input:

```bash
SGLANG_RESIDENCY_CENSUS=1 SGLANG_TRANSIENT_CENSUS=1 SGLANG_RESIDENCY_CENSUS_DIR=$OUT/census
```

Both instruments as per §5.2: NVML FREE at 100 ms, seam census (on by
default), `hostmem_sample.sh`.

### The four questions M1 must answer, in writing

1. **Confirm the virtual reserve.** `reserved` overshoots the 5090's
   **32088.5 MiB** total by 921 MiB at s50's breach and by **1990 MiB at
   rest**. Record `PYTORCH_CUDA_ALLOC_CONF` / `expandable_segments` from the
   boot env so the next reader does not repeat the desk's first mistake of
   treating `reserved` as physical. Bookkeeping, not a blocker.
2. **S3 vs {S1, S2} — the question this boot exists for.** Across
   `weights_refill`, does **`allocated`** stay flat while `free` drops (S3: a
   VMM commit outside torch), or rise with it (S1/S2: a torch allocation)? One
   column, one window, decided. The census prints `alloc=` now.
3. **The left term of the identity.** At-rest TP free on rank 0, for M0 and
   M1. Is it ≥ **8079 MiB**? State the number even when the answer is plainly
   no — it is the input to option (c).
4. **Does the outlier recur?** Pool M1's rank-0 `tp_to_pp` transients with the
   existing 196 via `excursion_485.py census`. Report the new worst and the
   new n. A second 7055-class event turns "one anomalous flip" into a rate,
   which is a different problem with a different fix.

### Commands

```bash
cd /spinning/wt-desk-485 && export PYTHONPATH=/spinning/wt-desk-485/python
# P2: re-solve with the fed gate (CPU, no device)
CUDA_VISIBLE_DEVICES=99 python scripts/cert_485/certify_485.py flags   # sanity
# ...then the planner re-solve against $OUT/census, and record the admitted cut.

# Q4: pool the new window with the existing reference class
python scripts/cert_485/excursion_485.py census   --log s50=/spinning/evidence-631/s50/boot_gate_cut_280000.log   --log s51=/spinning/evidence-631/s51/boot_arm_280000.log   --log m1=$OUT_M1/boot.log

# C2' verdict
python scripts/cert_485/excursion_485.py judge   --window s50=... --window s51=... --window m1=$OUT_M1/boot.log
```

### Do NOT do in this ticket

* **Do not run the A/B depth probe.** It measures the throughput of a cut P2
  may reject, and §6a makes the number a property of the cut. It belongs in
  P3.
* **Do not report M1 as certification window 2.** It is characterisation.
* **Do not argue "re-baseline makes s50 obsolete."** §2 shows the source is
  unchanged on this line.
* **Do not treat a clean corridor as certification.** s51 was clean and had
  the thinner margin.

---

## 8. Teardown and restore

1. Stop the corridor and hostmem samplers; confirm both files end with a
   complete line — a truncated series has no minimum.
2. Confirm 3 ranks alive and the window log carries its end marker.
3. Park artifacts under `/spinning/evidence-631/<shift>/`: both corridor CSVs,
   `hostmem.csv`, seam census, the arm/ship JSONs, boot logs, and the
   `certify_485.py judge` output verbatim.
4. Stop the heartbeat **before** releasing `/spinning/gpu-arb/holder`.
5. Restore serving if this window stopped it — whoever stopped it owns bringing
   it back — and verify with a real generation, not `/health` alone.

## 9. Desk validation already done (CPU, this branch)

| Item | Result |
|---|---|
| `certify_485.py smoke` | **7/7** red-on-demand: breach caught, one-window refused, margin-inside-spread refused, margin-clearing-spread certified, moving binding rank refused, thin `n_scored` refused, cache-hit sample refused |
| `certify_485.py flags` | **8/8 PASS** against `cb8da83774`, by building the real parser |
| `certify_485.py judge` on real s50+s51 | Reproduces the known state: s50 DIRTY (breach 669, seam agrees, 2 of 3 ranks), s51 CLEAN (+331 MiB) |
| `certify_485.py ordering` on real s50 | **The §2 finding, reproducible**: `breach precedes the event by 289 s -- it CANNOT be an artifact of that event` |
| `ruff check scripts/cert_485/` | clean |

## 10. Revision 2 — what was added at the desk, 2026-08-14

| Item | Result |
|---|---|
| `excursion_485.py smoke` | **12/12** red-on-demand, incl. the census's new `alloc=` format and backward compatibility with every artifact written before it |
| `excursion_485.py census` on real s50 + s51 | 196 flips; transient body 5654–6008, one at 7055; `weights_refill` step takes exactly 3 values |
| `excursion_485.py decompose` on s50 @ 11:44:06 | one stage differs, by 1258 MiB |
| `excursion_485.py judge` on real s50 + s51 | `NOT CERTIFIED (C2')`, −354 / −765 MiB |
| T1 gate producer + the pinned refusal | `test_seam_staging_producer_485.py` **20 passed**; the gate ADMITS `40,12,12` at seam 0.0 and **REFUSES** it at the measured 5800, short by 5425.1 MiB |
| CAN-FAIL PROOF for T1/T3/census format | sources reverted to the line: **14 failed / 6 passed** |
| Re-solve fork mapped (§5.1a) | two proxies bracket `42,11,11`-feasible and nothing-feasible; pool is **not** a recovering lever under the tight proxy |
| Regression | managers **590 passed / 0 failed**; planner **2380 passed / 6 failed**, all 6 identical with and without this branch |

No GPU, no serving process, no model touched. The copy in
`/spinning/wt-485-cert` was **not** modified.

Reproduce the reference-class test in one command:

```bash
python scripts/cert_485/certify_485.py ordering \
  --corridor /spinning/evidence-631/s50/corridor_planner.csv \
  --card gpu1_free --event-utc 11:48:55
```

No GPU, no serving process, no model touched.
