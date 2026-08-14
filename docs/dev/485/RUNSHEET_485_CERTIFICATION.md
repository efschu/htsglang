# RUNSHEET — #485 planner-cut certification, windows 2 and 3

**Purpose.** Close the open gate on `--pp-solve-cut`: *"2nd + 3rd clean
certification windows"* before a default-flip proposal goes to the user. Two
windows, one runsheet, criteria pre-registered in code so they cannot be
adjusted afterwards to fit the result.

**Derived from** `integration/r2` @ `cb8da83774` (re-resolved 2026-08-14; the
line had advanced from `9cedf43811`). Branch `chore/ticket-485-cert`,
worktree `/spinning/wt-485-cert`, `PYTHONPATH=/spinning/wt-485-cert/python`.

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

---

## 1. What IS in doubt — and it is a variance, not a mean

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
* **C2** `margin > spread`, where `margin = min(binding_min) - 1024` and
  `spread = max(binding_min) - min(binding_min)`. This is the direct encoding
  of "the margin is inside the spread": **a margin smaller than the movement
  already observed is not a margin.**
* **C3** the binding rank is the **same card** in every window. If it moves,
  the spread is computed over two different quantities and the windows are not
  repeats.

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

Reproduce the reference-class test in one command:

```bash
python scripts/cert_485/certify_485.py ordering \
  --corridor /spinning/evidence-631/s50/corridor_planner.csv \
  --card gpu1_free --event-utc 11:48:55
```

No GPU, no serving process, no model touched.
