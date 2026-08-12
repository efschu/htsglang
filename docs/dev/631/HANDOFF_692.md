# HANDOFF 692 — #485 taken to metal, merged, and deliberately NOT wired

Successor 48. Branch `feat/pp-family-cut-485`, rebased onto the flip line at
`3be93fa943`, measured HEAD `e645aa70c0`. Evidence:
`/spinning/evidence-631/s485/` (`RESULTS.md` is the verdict, `CALIBRATION.md`
the memory work, `README.txt` indexes the rest).

---

## 1. ERRORS FIRST

### 1a. The control arm in HANDOFF_485_PPCUT §3a is not the ship cut

§3a gives arm A as `--pp-stage-ratio 15,9,8` and §1d states "The ship config
is `[32,16,16]` (attention 8/4/4)". The LIVE ship boot I inherited runs
`--pp-stage-ratio 14,10,8`, which `derive_pp_layer_split` turns into
**`[28,20,16]`, attention `[7,5,4]`**. Running §3a's arm A would have measured
the planner cut against a cut nobody ships.

Knock-on: arm B is specified as "decoupled, KV-neutral" at attention `[8,4,4]`.
Against the REAL ship cut that moves an attention layer from rank1 to rank0,
so it is not KV-neutral and its "measure it on the corridor, not the clock"
rationale does not survive. B was dropped (§4).

The lesson is cheap and general: the parallel strand read the ship cut out of
`PROD_BRINGUP_BENCH.md`, not off the running process. I read it off
`/proc/<pid>/cmdline` and recomputed the split with the branch's own function.

### 1b. The acceptance criterion cannot be met as written: no per-rank compute/wait under PP

The brief (and §3b) require "ms/round per rank COMPUTE vs WAIT via
CollectiveClock". That instrument does not exist on this configuration.
`SchedulerMetricsReporter._install_rank_prefill_timer`
(`scheduler_components/metrics_reporter.py:379-382`) returns early when
`server_args.pp_size != 1`, and `RankPrefillLog`'s own docstring says why: PP
processes prefill results on the last stage only, which breaks the FIFO
pairing the timer needs. There is no `wait` number to report and inventing one
would be a fabrication.

Measured the **pipeline makespan** instead — which is the quantity the +27.6 %
prediction is actually stated in (§3's table is makespan). Anyone re-running
this should not go looking for the split; it has to be BUILT for PP first.

### 1c. Two hypotheses I was told to hold were refuted by measurement

* The brief: "treat per-card calibration as the default hypothesis … if the
  4982-vs-7582 gap reproduces, same-model cards are not interchangeable."
  It reproduces in direction and the conclusion is **wrong** — the residual
  follows the last-stage ROLE. One boot with `--rank-gpu-id 0,2,1` settles it
  (C35). The cards are interchangeable to the megabyte.
* §1e: "the residual is not cut-invariant, so a single scalar may not be the
  right shape." It **is** cut-invariant; a per-rank scalar is exactly right.

### 1d. My own errors

1. I sized the A/B at pool 340000 from the calibrated gate, and arm C wedged
   there. Redoing the control at 280000 cost a boot (~9 min). The gate was not
   wrong about residency — it has no seam-staging term (C34, law 23) — but I
   should have priced the flip's staging before choosing the pool, since the
   ship env sets `SGLANG_SEAM_ENTRY_MARGIN_MIB=512` in plain sight.
2. Building the flip-off argv, I removed only `--enable-phase-flip`. The code
   refused twice, correctly and informatively — first because
   `--phase-flip-purity` is inert without it, then because PP + speculation
   requires the flip. Two dead boots, ~6 min. Both refusals are good design;
   the error was mine for editing an argv by name instead of by family.

### 1e. Register collision, resolved

The #485 strand independently assigned **C29** and **law 21**, both already
used by the flip line (C29 = restore margin, law 21 = instrument floor). At
merge its entries were renumbered to **C33** and **law 22**, cross-references
fixed in `HANDOFF_485_PPCUT.md §5`. This shift's own entries are **C34**,
**C35** and **law 23**. One numbering, no duplicates.

---

## 2. THE RESULT

Full tables in `/spinning/evidence-631/s485/RESULTS.md`.

**The cut is real and larger than predicted.** Depth 179200, pool 280000, four
arms differing only in the cut:

| arm | layers | attn | wall s | vs control |
|---|---|---|---:|---:|
| A control (ship cut) | 28,20,16 | 7,5,4 | 95.436 | — |
| **C planner-optimal** | 42,11,11 | 10,3,3 | **63.246** | **+50.9 %** |
| D falsifier | 16,24,24 | 4,6,6 | 119.271 | −20.0 % |

Desk predicted +27.6 % and −50 %. **Direction right on both, magnitude wrong
on both, in opposite directions** — the roofline model is a usable ranker and
a bad estimator. Floor: A-vs-A 0.12-0.97 %, and **0.09 % across two separate
boots at two different pools**, so the separation is ~50x the noise. The
falsifier is measurably worse, so the instrument is not one-sided.

**And it cannot be booted in the shipped configuration.** With the flip on,
every cut that moves the attention split off `[7,5,4]` starves the seam
staging (short by 55 MiB in arm C) and the policy retries without backoff —
528 abandons — until the detokenizer heartbeat dies. The instance prints
"fired up and ready to roll" and never answers `/health` again, with every
scheduler stack IDLE in a normal wait. A second, distinct failure kills arm C
at a pool where staging fits: a pool-accounting invariant
(`available + withheld` over-counts `total` by 12783 tokens).

**Verdict: GATE, do not wire.** The condition in the brief was "wire if the
cut wins and the gate is calibrated". Both hold, and wiring would still be
wrong: the calibrated gate declared arm C feasible with 2617 MiB to spare and
was right about residency, while the flip needed 4881 MiB of TRANSIENT
staging the model has no term for. `solve_pp_cut` and `validate_pp_cut` stay
uncalled; `--pp-attn-stage-ratio` remains the manual surface.

---

## 3. CONFIRMATION WINDOW

Ship config restored from MY merged tree (`PYTHONPATH=/spinning/wt-485-ppcut/
python`), so the window doubles as the merge's acceptance test — the #485
commits must be inert when the new flags are absent, and this is that claim on
metal rather than in a unit test. Numbers in `WINDOW.txt`.

21 minutes, 9515 NVML samples, 426 seam-census troughs, **0 breaches on both
instruments**, 142 flips with 0 abandoned, 0 tracebacks, soak ok=118 err=0.
The restored boot runs the MERGED tree with no new flags and behaves exactly
like the ship boot did — which is the "byte-identical when absent" claim on
metal, not in a unit test.

**The watch item: the trend does not continue.**

| axis | N46 | N47 | **N48** |
|---|---:|---:|---:|
| gpu0_free MIN | 1435 | 1397 | **1477** |
| gpu1_free MIN | 2388 | 2061 | **2416** |
| gpu2_free MIN | 1713 | 1727 | **1725** |
| deepest seam trough | 1434 | 1396 | **1476** |
| soak ok / err | — | 72 / 0 | **118 / 0** |
| soak prefill tokens | — | 355993 | **554151** |

Above BOTH predecessors on gpu0, gpu1 and the deepest trough; above N46 on
gpu2 and 2 MiB under N47's there, i.e. equal within one sample. It did that
under **1.56x N47's prefill load**, the direction that would have DEEPENED a
real drift. The axis that fell furthest under N47 (gpu1, 2061) came back as
the highest of the three.

This supports N47's own hypothesis — load state, not degradation — but does
not prove it, because my soak shape is not identical to N47's either. The
defensible statement is the negative: **three consecutive windows do not form
a monotone decline.** C7 applies; these minima read a state. Not carried
forward as a drift; carried forward as a method fix — **compare on
load-matched windows or not at all.**

---

## 4. WHAT I DID NOT DO

* **Arm B was not run.** It is the pure-decoupling, KV-neutral arm and its
  specification is stale anyway (§1a). Re-specify it against `[7,5,4]` — the
  16 layer splits that hold that attention split are `[28,20,16]` through
  `[31,17,16]` — and judge it on the corridor, not the clock.
* **Cut versus token vector is not disentangled.** I set
  `SGLANG_UNEVEN_TOKEN_VECTOR` per arm to the arm's attention split so the KV
  arena follows the attention layers. Both arm C failures are therefore
  reachable from either the cut or the vector. **One boot decides it**: arm C's
  cut with the ship vector `7,5,4`. Do this first — it is cheap and it changes
  what C34 is about.
* **No per-rank compute/wait split** (§1b) — it does not exist under PP.

## 5. NEXT SHIFT, IN ORDER

1. The one-boot confound test above.
2. Add `seam_staging_mib` to `RankResources` and subtract it before the
   headroom objective. This is the work that converts a measured +50.9 % from
   a locked prize into a reachable one, and it is the highest-value follow-up
   on this ticket by a wide margin.
3. The flip's retry loop needs a backoff and a refusal: 528 identical
   abandons that starve the heartbeat should be one refusal and a clear death,
   not a server that reports itself ready and then goes silent. This belongs
   to the crash strand (#622/#649) as much as to #485 — it is a NEW wedge
   shape, reached without any spill or cutover fault.
4. The pool-accounting over-count under a skewed token vector.
5. Arm B on the corridor axis.
