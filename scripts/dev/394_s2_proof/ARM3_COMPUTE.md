# Arm 3 — link-proportional expert COMPUTE placement (#394 slice 3 / task #439)

The two arms this directory shipped with move BYTE ownership. Arm 3 moves the
assignment: which rank executes which expert. It is the arm ANALYSE_393 §7.3
called Path A′, and it is the first arm in this window whose predicted per-rank
H2D delta is NOT null.

    ARM=equal          baseline, pre-#394 plan field for field
    ARM=proportional   slice 2: cold BYTES follow the links, compute does not
    ARM=compute        slice 3: cold COMPUTE follows the links   <-- this file
    ARM=compute-cal    slice 3 with the traffic coefficients measured on arm 1
                       -- FALSIFIED 2026-08-03 on two legs, end-to-end and
                       mechanism; see "What the confirmation window measured"

**Status in one line (2026-08-03).** The arm served tokens in two windows and the
green-corridor re-proof PASSED all five gates: `compute` measures **1.4307x** on
the transfer term and **-6.42 %** end-to-end against a same-window floor of
0.424 % spread, with per-card corridor minima of 655-1318 MiB against the
400 MiB floor. That figure is the FINAL, WORK-MATCHED dump revision — the only
basis on which two arms may be divided by each other (see "Which revision to
read"). The night window re-reads to **1.4253x** on the same basis, so the two
windows agree to 0.4 % and the model's predictions (1.427x green, 1.411x night)
sit between them. The uncalibrated solve is the recommendation and the flag's
plain symbol; `compute-cal` stays falsified on two legs, end-to-end and
mechanism. Evidence:
`/spinning/gpu-battery-results/2026-08-03_439_green/RESULTS.md`.

## The one flag

    --rank-moe-ratio link              # the shipped solve, uncalibrated
    --rank-moe-ratio link-calibrated   # EXPERIMENTAL, falsified model

Under the #82 GGUF expert-dim shard the "moe" family vector IS the expert
range: rank `r` owns `[lo_r, hi_r)`, foreign topk ids remap to a zero padding
expert, and the existing TP all-reduce sums the disjoint contributions
(`fused_moe_triton/layer.py`, `__init__` + `forward_local`). So no new mechanism
is involved — arm 3 is arm 1 with a different vector, solved rather than
inherited from the VRAM plan.

The solve (`layers/moe/expert_compute_placement.py`) holds each rank's
GPU-RESIDENT expert mass at exactly what the base plan gives it — arm 3 is
VRAM-neutral, so the reserve, the ledger and the corridor are the ones arm 1
was validated at — and redistributes only the STREAMED remainder in proportion
to the measured link weights:

    resident_r = f_r * b_r                      # fixed
    share_r    = resident_r + (1 - sum resident) * normalise(l_r / c_r)

`l` comes from the same #394 provenance chain arm 2 uses (env > card-probe H2D
> NVML nameplate > refusal; `absent` is refused, never guessed). `c` is the
per-rank cold-traffic coefficient, and under `link` it is 1.0 on every rank —
the symbol decides, not an environment variable. `link-calibrated` is the only
way to spend measured coefficients, it requires
`SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS`, and plain `link` REFUSES while that
variable is set rather than quietly solving something else (#458). Before that
split, a coefficient export left over from one arm silently turned the next arm
into the calibrated one with the command line still reading `link`.

## NOTE — VRAM neutrality, and the fixed point (#439, 2026-08-02)

The sentence above said "by construction" and the SOLVE honoured it. The
RUNTIME did not, and the 2026-08-02 battery died of it: residency is sized as
`R = resident_slot_count(E, f)` with `E` the rank's OWN expert count, and the
installed vector is exactly what changes `E`. tp2 went 71 → 85 experts, its
resident mass rose 19.5 %, and a 20 GiB 3080 that the baseline already left
~515 MiB free OOM'd in `_ensure_tiers -> allocate`.

The obvious correction — lower the resident FRACTION until the mass comes
back — is circular, because the solve reads the fractions as an input. The
battery measured the loop: rescaled fractions `0.4812,0.5296,0.3516` returned
`161,91,113` instead of `160,79,119`, a placement for which the correction just
computed is already wrong.

**Decision: break the loop by construction, do not iterate it.**

* The SOLVE keeps reading the fractions the OPERATOR set, against the BASE
  plan. Its inputs are untouched, so the vector is solved once, from the rig.
* The RUNTIME holds residency at the pre-link baseline: rank `r` keeps exactly
  `resident_slot_count(base_extent_r, f_r)` slots, sized off the base plan the
  launcher publishes on `SGLANG_MOE_COMPUTE_BASE_PLAN`.
* That correction is expressed as a DERIVED per-rank fraction on a channel the
  solve never reads (`resident_fraction_held_at_base_plan`). Nothing feeds
  back; there is no second solve and no fixed-point iteration.

`--rank-moe-resident-fraction` therefore keeps meaning exactly what it meant —
the operator's fraction of the BASE shard — which is also what keeps the
treatment arm comparable with the baseline arm that shares it. Exact (slot for
slot) under the #82 expert-dim shard; nearest-integer on the width-sharded MoE
path, where a slot is indivisible and no exact representation exists. Pinned in
`test_expert_compute_placement_439.TestResidencyIsHeldAtTheBasePlan`, whose
can-fail arm reproduces the 31 → 37 slot inflation on tp2.

## Resolution point, and why it is not in the worker

The launcher resolves the symbol once, after it publishes the rank -> card
vector and before the spawn loop; the workers inherit the numbers in the pickled
`ServerArgs`. Three workers re-deriving the vector from three independent NVML
reads would put the group's expert COVERAGE on the outcome of a race — a hole or
an overlap in the ranges is a silently wrong all-reduce, not a hang. A symbolic
value that reaches a worker anyway is a hard error there
(`scheduler.uneven_family_plans`), never a fall-back to the base plan.

## Predicted numbers — READ THE BASE PLAN FIRST

The table below is keyed to a base plan of `400,256,344`. **The reference
recipe does not produce that plan.** `--rank-tp-ratio auto` on the 2026-08-02
battery resolved to `30407,19080,19080` (shares 0.44346 / 0.27827 / 0.27827),
and every number keyed to `400,256,344` — the vector, the H2D, the 1.358x /
1.584x band — is a prediction for a rig configuration that boot did not have.
The numbers for the plan the recipe actually resolves are in "What the
confirmation window measured" below — and those are MEASURED now, not
predicted. Read the resolved plan off the boot log, as this file has always
said, and use the section that matches it. Everything in the next two sections
is a worked example on a plan no boot has run; it is kept because it is where
the model is derived, not because it is a target.

## Predicted numbers for the reference recipe

Inputs, all from `docs/dev/BENCH_394_v4flash_club3090.md` (V4-Flash UD-IQ3_XXS,
TP=3, bench-length generations) and the arm-1 launch:

| rank | card / slot | base share | resident fraction | measured link | measured H2D | implied transfer |
|---|---|---|---|---|---|---|
| tp0 | 5090, x8 | 0.400 | 0.485 | 14.42 GB/s | 7502 GiB | 559 s |
| tp1 | 3080, **x4** | 0.256 | 0.42 | 6.45 GB/s | 5155 GiB | **858 s** |
| tp2 | 3080, x8 | 0.344 | 0.42 | 13.41 GB/s | 5181 GiB | 415 s |

`ARM=compute` (uncalibrated, `c = 1,1,1`) solves to

    --rank-moe-ratio 123,61,104        (shares 0.4270 / 0.2118 / 0.3612)

| readout | prediction |
|---|---|
| per-rank H2D (GiB) | 8487 / 3619 / 5628 (was 7502 / 5155 / 5181) |
| implied transfer (s) | 632 / 602 / 451 (was 559 / 858 / 415) |
| clock (slowest rank) | 858 -> **632 s = 1.358x** |
| group H2D | ~unchanged (17838 -> 17733 GiB) |
| which rank is the clock | tp1 -> **tp0** |
| `moe_compute_policy` | `link-proportional` |

That is the 1.36x class ANALYSE_393 predicted for the short-probe mix. It is
the number to hold arm 3 to ON THIS BASE PLAN, which is not the one the recipe
resolves — see "What the confirmation window measured". It falls short of BENCH_394's 1.54x "ideal
proportional placement" for a reason that is measured rather than hand-waved:
the first-order model says a rank's H2D share is `b_r (1 - f_r)`, i.e.
37.2 / 26.8 / 36.0 %, and arm 1 measured 42.1 / 28.9 / 29.0 %. The ranks re-fetch
at materially different rates (hit rates 0.772 / 0.843 / 0.841), so equalising
the MODELLED cold mass does not equalise the MEASURED bytes.

`ARM=compute-cal` closes that gap by feeding arm 1's own per-rank H2D back in
(`cold_traffic_coefficients_from_measurement` -> `1.1251, 1.0726, 0.8023`):

    --rank-moe-ratio 176,90,181        (shares 0.3938 / 0.2012 / 0.4050)

| readout | prediction |
|---|---|
| per-rank H2D (GiB) | 7275 / 3254 / 6765 |
| implied transfer (s) | 542 / 542 / 542 |
| clock | 858 -> **542 s = 1.584x** |
| `moe_compute_policy` | `link-proportional-calibrated` |

**That prediction is the one hardware overturned — but not on the transfer
term.** Running both sub-arms is what turned the band into a measurement. On the
pre-teardown dump revision the calibrated arm read 1.439x against 1.496x, and
that comparison REVERSES once both arms are read at the same work point:
work-matched, `compute-cal` reads **1.4573x** against `compute`'s **1.4253x**.
The transfer term is therefore not something the rejection can rest on. What
overturned the calibrated arm is its end-to-end result and its mechanism — the
next section — and that is why `compute-cal` is a separate, experimental symbol.

## What the confirmation window measured (2026-08-03 night window, RAN)

Referred to as the "night window" throughout this file, after its own record
(`2026-08-03_439_confirm/RESULTS.md`: "2026-08-02/03, night window 2").

Three boots, `equal` / `compute` / `compute-cal`, DeepSeek-V4-Flash UD-IQ3_XXS,
TP=3, 900 tokens x 3 runs x 1 warmup, `--disable-cuda-graph`. Full record:
`/spinning/gpu-battery-results/2026-08-03_439_confirm/RESULTS.md`. All four
gates PASS. Rank order verified from preflight's NVML table, not assumed:
**tp0 = 5090 x8 (14.42 GB/s), tp1 = 3080 negotiated x4 (6.45 GB/s), tp2 = 3080
x8 (13.41 GB/s)**.

| arm | vector | h2d GiB tp0/tp1/tp2 | clock | speedup | ms/token | vs equal | above the floor? |
|---|---|---|---|---|---|---|---|
| `equal` | 30407,19080,19080 | 1806.2 / 1157.6 / 1143.0 | tp1, 192.7 s | 1.000x | 138.2 | — | — |
| `compute` | 160,79,119 | 1729.7 / **672.7** / 1455.5 | tp0, 128.8 s | **1.496x** | **127.6** | **-7.67 %** | **YES** |
| `compute-cal` | 164,85,130 | 1716.9 / 730.2 / 1672.1 | tp2, 133.9 s | 1.439x | 136.9 | -0.94 % | NO |

The H2D and speedup columns above are the PRE-TEARDOWN dump revision, which is
what that window quoted. Read at a common work point instead, the same window
gives `compute` **1.4253x** and `compute-cal` **1.4573x** — see "Which revision
to read". The ms/token and floor columns are probe-measured, not dump-derived,
and are unaffected by the revision choice.

Same-window A-vs-A floor, measured in the `equal` boot: **CV 2.12 %, spread
4.09 %** (900 tokens; the prior window's 1.19 % / 2.35 % at 450 tokens does not
transfer). The clock rank moved OFF tp1, which is the mechanism's whole point.

Three findings came out of it, and the rest of this file is what they changed.

**Finding 1 — the arm works and the uncalibrated solve is the recommendation.**
`compute` is the only arm whose end-to-end effect clears the floor: **-7.67 %**
against a 4.09 % spread. On the work-matched revision its transfer term is
**1.4253x** against a re-derived prediction of 1.411x. The recurring "it beat
its own prediction by 6 %" reading came from the pre-teardown revision and does
not survive work-matching; the prediction was right. The DESK-WRITTEN label
lifts for the slice-3 compute path.

**Finding 2 — the calibration is falsified, and the rejection rests on exactly
TWO load-bearing legs.**

**Leg 1 — end-to-end. This is the economically decisive leg.** `compute-cal`
measured **-0.94 %** ms/token against the baseline, inside that window's own
**4.09 %** A-vs-A spread: a non-result. The end-to-end figure is probe-measured,
not dump-derived, so no choice of dump revision can move it.

**Leg 2 — mechanism, via the falsifier this file named in advance** (readout
item 4: *"if the per-rank hit rates move a lot when the ranges move, the
coefficient is not a property of the rank"*). They move a lot:

| rank | `equal` | `compute` | `compute-cal` | expert extent |
|---|---|---|---|---|
| tp0 | 0.7637 | 0.7618 | 0.7774 | 115 -> 115 -> 112 (stable) |
| tp1 | 0.8450 | **0.9050** | 0.9025 | 72 -> 58 -> 58 (**+7.1 %** as the range SHRANK) |
| tp2 | 0.8474 | **0.7972** | **0.7814** | 72 -> 86 -> 89 (**-7.8 %** as the range GREW) |

The hit rate tracks the SIZE of the owned range, because a smaller range fits
the cache better. The calibrated solve read tp2's below-average coefficient
(0.9586) as spare capacity, moved three more experts onto it, each cost more
traffic than modelled, and tp2 became the new clock. The green-corridor window
reproduced this on independent data with a clean control: tp0's range did not
move and its hit rate did not move (0.7636 -> 0.7649), tp1's range shrank
72 -> 57 and its hit rate rose 0.8465 -> 0.9092, tp2's grew 71 -> 86 and its hit
rate fell 0.8415 -> 0.7915.

**Demoted, and it is NOT a leg: the transfer-term comparison.** The sentence
"it reached only 1.439x against 1.496x" divided two counters sampled at
different fractions of their runs. Work-matched, the same window gives
`compute-cal` **1.4573x** against `compute`'s **1.4253x** — the calibrated arm
slightly WINS that term. The rejection never depended on it and does not cite it
any more.

**Demoted with it: "it underperformed its own 1.498x prediction."** That
statement is the same pre-teardown ratio compared against a prediction, so it
inherits the same defect and is not evidence either.

Registered in `planner/rejected.py` (`moe_link_calibrated_coefficients`,
NOT_DEFAULT) and demoted to its own symbol; see "The one flag".

**Finding 3 — `--rank-auto-reserve-mib auto` is INFEASIBLE for this recipe**, and
this file's own spec was internally inconsistent about it. `auto` derives
3968 MiB uniformly (`512 + max(chunked_prefill, 2048) * 1.5 + tp * pp / 8 * 1024`;
the graph term is correctly 0 under `--disable-cuda-graph`), leaving budgets
`28639,16512,16512`, and the boot is refused after weight load:

```
ValueError: The per-rank budget leaves no GPU memory for the KV cache under
--rank-gpu-memory-mib on rank 1: the 16512 MiB (16.12 GiB) budget is spent on
weights + runtime state 17.59 GiB -- 17.59 GiB together, 1498 MiB more than the
budget, before a single KV token.
```

Gate 4 demands the base plan `30407,19080,19080`, and that plan IS
`32607-2200 / 20480-1400 / 20480-1400` — produced by the pinned reserve
`2200,1400,1400` and by nothing else. So `auto` could not have satisfied Gate 4
even if it had booted, and **the Gates below pin the reserve.** `auto` is not
taught to avoid this, because it cannot be: it sizes the reserve from the
activation heuristic and never sees the checkpoint, and the weight bytes it
would need depend on the shard ratio, which is derived FROM these budgets. What
#458 changed is the refusal — the error now names the derivation, says `auto`
cannot self-correct, and gives the pinned value that fits
(`ServerArgs.derived_reserve_infeasible_note`, pinned by
`test_uneven_tp_args.TestDerivedReserveInfeasibility`).

## Which revision to read — THE RULE, and it replaces the old one

**Read the FINAL, WORK-MATCHED dump revision. Never divide two arms' pre-teardown
snapshots by each other.** The previous instruction in this file — *"quote
`read_arm.py`'s pre-teardown numbers, never the post-SIGTERM `expert_stats_*.json`
revision"* — is WITHDRAWN. It is the defect behind every inflated transfer-term
ratio this ticket has reported.

Why. Each rank writes its dump on its own 45 s timer, so a pre-teardown read
catches an arm at whatever fraction of its run the last tick happened to land
on. Within one arm the three ranks are well synchronised; the two ARMS are not.
In the green-corridor window `equal` was caught at 96.8 % of its run and
`compute` at 91.9 % of its own, and the treatment arm's accumulating H2D counter
was therefore read ~5 % early, inflating the ratio by about that much:

    work-matched   equal 199.3 s -> compute 139.3 s = 1.4307x   (prediction 1.427x)
    pre-teardown   equal 192.8 s -> compute 128.3 s = 1.5028x

The final revision has no such problem: it is a common, well-defined endpoint,
and the two arms are work-matched to 0.05 % (163486 vs 163572 tokens, 155359 vs
155445 forwards, 980916 vs 981432 activations — and within each arm all three
ranks are identical). It is the right basis for a ratio.

This retro-corrects the night window as well. Re-read on the work-matched basis:

| window | as reported (pre-teardown) | work-matched | prediction |
|---|---|---|---|
| `2026-08-03_439_confirm`, `compute` | 1.496x | **1.4253x** | 1.411x |
| `2026-08-03_439_green`, `compute` | 1.5028x | **1.4307x** | 1.427x |

The two windows agree to 0.4 % work-matched and the model's prediction sits
between them.

Operationally: run `read_arm.py` AFTER teardown, and take every cross-arm number
from `--against`:

    python3 read_arm.py <run> equal --against compute --links 14.42,6.45,13.41

**The rule is enforced there, not by the reader (#523).** Until #523 this file
said "compare the two `work=` lines before dividing" and the tool printed them
and stopped. That is the check both #439 windows skipped, and it is why 1.5028x
and 1.496x were published. `--against` is now the ONLY path in this directory
that produces a cross-arm number — per-rank H2D delta, group delta, transfer
term, speedup — and it REFUSES with a named reason, a non-zero exit and no
number at all when:

| reason | what it caught |
|---|---|
| `non-final-revision` | the ranks of one arm disagree: a pre-teardown read, each rank on its own 45 s tick |
| `work-mismatch` | the two arms are internally final but sit at different work points (default tolerance 0.5 %, below the window's own A-vs-A floor) |
| `missing-counter` | a dump revision from before the work counters — the work point is unknown, so the ratio is unfalsifiable |
| `rank-count-mismatch` | the arms are not the same group |
| `link-count-mismatch` | the link vector does not cover every rank, so there is no clock |

The single-arm readout is unchanged and still prints `work=`; what it can no
longer do is be the input to a hand-computed ratio, because it never prints one.

A note on the quoted figure: **1.4307x** is `199.3 / 139.3`, the two clock
seconds as this file displays them to one decimal. From the full-precision
`h2d_bytes` the same dumps give **1.4304x**, which is what the tool prints. The
0.02 % between the two readings is a quoting artifact, 20x under this window's
own 0.424 % A-vs-A floor; both are pinned in the #523 test so neither can drift.

## Corridor: BREACHED at the measured recipe, and the arithmetic that repairs it

The window sampled per-card free VRAM every 5 s across the whole serving window
(`corridor_sampler.sh`, now in this directory). **Minima over the serving window,
not a post-boot snapshot:**

| arm | nvml0 = tp1 (3080 x4) | nvml1 = tp0 (5090) | nvml2 = tp2 (3080 x8) |
|---|---|---|---|
| `equal` | **215 MiB** | 1262 MiB | **215 MiB** |
| `compute` | **249 MiB** | 1288 MiB | **221 MiB** |
| `compute-cal` | **251 MiB** | 1282 MiB | **211 MiB** |

All three arms are outside the >= 400 MiB floor on both 3080s, identically in
kind, so the breach does not bias the A/B — but it does mean that window's point
was measured in a red corridor. Note what changed the verdict: the ~515 MiB this
file previously quoted was a single POST-BOOT `nvidia-smi` line (the window's own
post-boot readings were 549/1374/521), and the serving minimum sits 250-330 MiB
below it. **Judge the corridor at peak, never at idle.**

The arithmetic, and it is deliberately simple. Under `--rank-gpu-memory-mib` the
budget is `nvml_total - reserve` and the KV pool takes whatever the budget leaves,
so a MiB added to the reserve is a MiB the pool never allocates and the card
keeps free:

    worst 3080 minimum over all arms   = 211 MiB   (nvml2, compute-cal)
    corridor floor                     = 400 MiB
    deficit                            = 189 MiB
    +200 MiB reserve  ->  411-451 MiB free   (inside by 11 MiB: not green)
    +400 MiB reserve  ->  611-651 MiB free   (inside by >= 211 MiB: green)

**Repaired recipe: `--rank-auto-reserve-mib 2200,1800,1800`** (the 5090 keeps
2200 — its minimum was 1262 MiB and it was never near the floor). This is now
the `boot_ab.sh` default.

This is a NEW WINDOW, not a tweak, and the reason is Gate 4: the reserve moves
the budgets, the budgets ARE the derived weight plan, so the base plan moves with
it. Everything downstream follows:

| quantity | as measured (1400) | repaired (1800) |
|---|---|---|
| budgets = NVML total - reserve | 30407, 19080, 19080 | 30407, **18680**, **18680** |
| base plan (`--rank-tp-ratio auto` derived weights) | 30407,19080,19080 | **30407,18680,18680** |
| base expert counts of 256 | 114 / 71 / 71 | **115 / 71 / 70** |
| solved `--rank-moe-ratio link` vector | 160,79,119 | **213,104,157** |
| installed expert counts | 115 / 58 / 86 (+1 pad each) | **115 / 56 / 85** |
| predicted transfer-term speedup | 1.411x (work-matched 1.4253x) | **1.427x** (work-matched **1.4307x**) |

(Solved with the module's own functions against the same measured links
14.42/6.45/13.41 GB/s and fractions 0.485,0.42,0.42; the prediction chain is
RESULTS.md §7's, re-run on the new base plan. The band set for the next window
was **1.43x - 1.51x**, widened upward from the apparent 6.0 % prediction beat —
which the work-matched re-read shows was an artifact of the dump revision, not a
margin. Both windows land near the band's lower edge, which is where the model
put them.)

The repaired column is a prediction and the green-corridor window measured it
slightly differently: base extents came out **116 / 72 / 71** and installed
extents **116 / 57 / 86** (conserved, 259 either way), against the 115/71/70 and
115/56/85 predicted here. The vector `213,104,157` and the base plan
`30407,18680,18680` are exactly as predicted.

**The off-by-one, reconciled (#523).** It is not a measurement difference and
neither number is wrong: they are two readings of the same partition, and they
differ by the **#82 zero padding expert**, exactly one slot per rank.

    partition_units(256, [30407,18680,18680]) = 115, 71, 70   sum 256
    partition_units(256, [213,104,157])       = 115, 56, 85   sum 256

The predicted table above counts REAL experts, which is why it sums to
`num_experts`. The LOGGED extent counts the pad slot as well: the residency
correction builds its counts as `partition_units(...)[rank] + 1`
(`expert_compute_placement.py:708-709`, and the reason is stated at :681-682 —
foreign top-k ids remap onto that expert and it is resident on every rank), so
Gate 3's table reads 116 / 72 / 71 and 116 / 57 / 86 and sums to 259 = 256 + 3.
Nothing to do with a warmup discard or a teardown boundary; the extents are not
a counter at all.

Both readings are load-bearing and neither may be "corrected" into the other:
drop the pad from the logged reading and the resident-mass arithmetic that
consumes it loses a slot per rank; add it to the predicted reading and the
extents stop summing to `num_experts`. Both are pinned in
`test/registered/unit/layers/moe/test_work_matched_counters_523.py`
(`TestTheExpertExtentOffByOne`), against the exact vectors of this window.

Second-order effect to watch, not to correct in advance: the 3080s' share of the
weight plan drops 0.27827 -> 0.27565 each, so ~0.5 pp of total weight mass moves
onto the 5090. Its serving minimum was 1262 MiB; a few hundred MiB of that is the
worst case, and it stays green. If it does not, that is a finding, not a reason to
re-tune mid-window.

## Green-corridor window — RAN 2026-08-03, PASS on all five gates

Record: `/spinning/gpu-battery-results/2026-08-03_439_green/RESULTS.md`. Outcome
against the gates below: corridor GREEN on every card in both arms (655 / 1318 /
1039 MiB `equal`, 663 / 1318 / 1007 MiB `compute`, 1 Hz, whole run); resolved
plan `[30407, 18680, 18680]` in both boot logs; `link-proportional` with vector
`213,104,157` and the three residency lines; A-vs-A floor CV 0.223 % / spread
0.424 %; transfer term **1.4307x** work-matched (1.5028x pre-teardown) and
**-6.42 %** end-to-end. The 1.4307x point is acceptance-evidence.

The ticket as it was written, kept because the gates are what the result is read
against:

One window, **two boots**: `equal` and `compute`. `compute-cal` is dropped — it is
falsified and registered, and re-running it would spend a card window
re-measuring a rejection. That is what makes this window cheaper than the last.

```
export RUN=/spinning/gpu-battery-results/$(date +%F)_439_green
export REPO_ROOT=/spinning/htsglang VENV=/spinning/htsglang-gpu/.venv
export WT=<the worktree> PORT=30439 RANK_GPU_ID=<from preflight.sh>
export RESERVE_MIB=2200,1800,1800          # the repaired recipe; NOT 'auto'
bash "$WT/scripts/dev/394_s2_proof/preflight.sh"        # must print PREFLIGHT OK
bash "$WT/scripts/dev/394_s2_proof/corridor_sampler.sh" "$RUN/corridor_equal.csv" 1 &
bash "$WT/scripts/dev/394_s2_proof/run_arm.sh" equal
touch "$RUN/corridor_equal.csv.stop"
bash "$WT/scripts/dev/394_s2_proof/corridor_sampler.sh" "$RUN/corridor_compute.csv" 1 &
bash "$WT/scripts/dev/394_s2_proof/run_arm.sh" compute
touch "$RUN/corridor_compute.csv.stop"
```

**Gates.**

1. **Corridor, and it is a GATE this time, not a note.** Sample at **1 Hz**
   during load (the 5 s of the last window biases the minimum high, and 1 Hz
   costs nothing). Per card, over the SERVING window only, minimum free VRAM
   **>= 400 MiB**. Expect ~611-651 MiB on the 3080s and >= ~900 MiB on the 5090.
   Map nvml columns to ranks through preflight's table — nvidia-smi order is not
   `--rank-gpu-id` order on this rig.
2. **Gate 4 (resolved plan).** `--rank-tp-ratio auto: derived weights
   [30407, 18680, 18680]` in both boot logs. A different plan means a different
   reserve reached the boot and the predictions below do not apply.
3. **Gate 3 (self-identification).** `facts_compute.txt` must show
   `moe_compute_policy=link-proportional` — NOT `-calibrated`, and not
   `base-plan` — plus `moe_compute_vector=213,104,157` and the three per-rank
   `resident fraction held at the base plan` lines.
4. **A-vs-A floor first**, in the `equal` boot, 900 tokens x 3 runs x 1 warmup.
   The expectation is the last window's CV 2.12 % / spread 4.09 %; a floor from
   another window does not cover this one.
5. **Speedup.** Transfer term `h2d_r / link_r`, clock = slowest rank, read off
   the FINAL WORK-MATCHED revision. Expected **1.43x - 1.51x** (prediction
   1.427x). End-to-end **-6 % to -9 %** ms/token, which must clear the floor
   measured in gate 4 to be reportable at all.

**Labelling rules, so the result is readable without the run dir.**

* A boot refused by the budget check is **INFEASIBLE-AT-RESERVE**, and the label
  carries the reserve, the derived budgets and the shortfall. It is not "the arm
  failed".
* A boot that serves but whose corridor minimum is below 400 MiB on any card is
  **CORRIDOR-RED**, and its speedup is recorded but **not acceptance-evidence**.
  That was the status of the night window's point; the green-corridor window
  cleared it.
* **No reserve-shrinking rescues.** If a card is short, the window ends and the
  finding is the shortfall. Lowering the reserve to buy a boot changes the base
  plan, hence the vector, hence what is being measured — and it silently converts
  a corridor breach into a different experiment.
* The reserve is ONE value for the whole window, identical on both arms. A
  reserve that differs between arms is a second treatment.
* Quote the FINAL, WORK-MATCHED dump revision: run `read_arm.py <run> A
  --against B` AFTER teardown, which is the only path that produces a cross-arm
  number and which refuses when the two arms did not do the same work. The old
  rule ("quote the pre-teardown numbers, never the post-SIGTERM revision") is
  WITHDRAWN — see "Which revision to read", where it is measured to inflate the
  ratio by ~5 %.

**Also read out this time:** the per-rank compute/wait split (see readout item 2
— grep the right string, the last window looked for the wrong one and reported
the instrument as silent).

## What must be read out, per rank

The slowest-rank rule governs every line: a group mean hides exactly the effect
this arm exists to move.

1. **Per-rank H2D** (`totals.h2d_bytes` in the #390 dump). The primary readout.
   Unlike arms 1 and 2, a NULL delta here falsifies the arm.
2. **Per-rank ms/round.** Transfer is part of a ~135 ms/token decode on this
   recipe, so the end-to-end effect is smaller than the transfer-term ratio.
   Report both, and report the per-rank split (compute vs wait) from the #252
   CollectiveClock — a rank that stops being the clock should show its wait time
   move, not vanish.

   **Grep for the LINE, not for the instrument.** The 2026-08-03 window reported
   "zero CollectiveClock lines in any boot log" and skipped this readout. The
   instrument was working the whole time: the string `CollectiveClock` is a
   class name and is never logged. The line is

       Prefill rank batch, #new-token: N, #cached-token: C, #chunks: K,
       gpu-ms: T (compute Tc, wait Tw)

   and that window's own `boot_equal.log` carries 15 of them, `boot_compute.log`
   18, on all three ranks (e.g. `TP0 ... gpu-ms: 4472.4 (compute 3113.4, wait
   1359.0)` next to `TP1 ... (compute 4347.5, wait 124.9)` — the same total, a
   3.5x spread in wait, which is the shard-imbalance signal). Read them with
   `scripts/gpu_battery/s12_log_analyse.py`, which already parses exactly this
   line and reports per-rank compute/wait medians and the wait share; do not
   re-implement the regex.

   **What the instrument genuinely does NOT cover, and it is a real limit of
   this readout:** the clock is armed only around PLAIN PREFILL forwards on the
   target runner (`metrics_reporter._install_rank_prefill_timer`; not the draft
   runners, not `pp_size > 1`, and the split is dropped entirely for a
   graph-covered forward because a replayed graph never runs the Python body
   that records the events). The headline of this arm is a DECODE measurement
   with prefill excluded by construction (`decode_probe.py:62`), so there is no
   decode-side compute/wait split to read at all. Report the prefill split as
   the shard-imbalance evidence it is, and do not present it as the decode
   round's split. Extending the clock to decode is a separate piece of work and
   is not in this window.
3. **`moe_compute_policy` + `moe_compute_vector`** (new in the dump). An arm
   whose policy reads `base-plan` ran the baseline; any delta against it is a
   delta between two baselines. This is the same self-identification rule
   `host_shard_reachability` enforces for arm 2.
4. **Hit rate, per rank** (`totals.hit_rate`, `unique_hit_rate`). This fills the
   row ANALYSE_389 left open and it is the input the calibrated sub-arm needs.
   It is also the falsifier for the solve's one modelling assumption: if the
   per-rank hit rates move a lot when the ranges move, the coefficient is not a
   property of the rank and the calibrated arm's prediction is wrong.
5. **Per-rank owned expert count.** Should match `partition_units(E, vector)`.
   A mismatch means the vector did not reach the layer.

## One trap, named

`SGLANG_MOE_HOST_SHARD_RATIO` does double duty: it is the strongest source of
the link weights AND, when equal, the switch that turns cold-expert delegation
off. The equal arm sets it to `1,1,1` for the second reason. **The slice-3 arms
must NOT**, because the first reason would hand the solve a uniform link
profile and turn the treatment into the baseline. They hold byte ownership at
the baseline by leaving `SGLANG_MOE_COLD_TIER_SHM` unset instead, which is the
switch that actually governs delegation. The solve logs a WARNING naming this
variable whenever it resolves to the identity, which is what that mistake looks
like from the boot log.

## Measurement discipline

* Bench-length generations (800-1000 tokens). The 2026-08-02 battery measured
  CV 1.0-1.4 % there against ~5 % on a 96-token probe: measurement length, not
  the rig, sets the floor. Every predicted delta above is far above 1.4 %.
* Same-boot A-vs-A floor before any A/B delta is quoted, per canon. First boot
  after a cache change is a JIT outlier.
* Same WORK POINT before any A/B ratio is quoted. Counters that accumulate over
  a run (H2D bytes above all) are only comparable between two arms at a common
  endpoint — see "Which revision to read".
* Arms in one window, same recipe, same reserve (`2200,1800,1800` since #458 —
  see the corridor arithmetic; the green-corridor window ran it and is green,
  `2200,1400,1400` is what the night window ran and it is corridor-red), same
  `--rank-moe-resident-fraction
  0.485,0.42,0.42`. The solve holds the resident mass fixed, so changing the
  fraction between arms changes the treatment.
* `--disable-cuda-graph`, as the published baseline for this configuration does.

## Corridor and preflight

VRAM-neutral by construction, so the corridor is arm 1's: per-card free
>= 400 MiB, no registered posting wasting > 1.5 GiB net. **Arm 1's corridor was
itself judged on a post-boot snapshot and is red at peak** — see the corridor
section above for the measurement and the repaired reserve. Host DRAM is where the
mass moves; total pinned bytes are unchanged, but the PER-RANK host pool grows
on tp0/tp2 and shrinks on tp1, so re-run `preflight.sh` for the `/dev/shm`
headroom if arm 3 is combined with `SGLANG_MOE_COLD_TIER_SHM=1`. Arm 3 does not
require the shared tier: it moves compute to the bytes rather than bytes to the
compute, and the two compose but are independent.

## Hazards this arm walks into, and what covers them

| hazard | why it applies | cover |
|---|---|---|
| uneven expert ranges x GGUF shard boundaries (#109 MMQ-OOB) | the vector changes every range | dim 0 has no quant constraint (layer.py `_gguf_expert_shard`); the solve is swept against `partition_units` for >= 1 expert, exact sum, gapless tiling |
| router/dispatch must stay rank-uniform (collective family) | the ranges must be disjoint and cover [0, E) | single resolution point in the launcher; pinned with the #431 recorder + a can-fail arm |
| combine correctness (#80) | the all-reduce sums disjoint partials | unchanged code path; ranges stay a partition |
| `moe.cuh` nrows binding (#112) | per-rank expert counts change | counts come from the same `partition_units` the layer uses |
| VRAM ledger drift | more experts on tp0 | resident mass held fixed; nothing in the ledger moves |

## Run order

```
export RESERVE_MIB=2200,1800,1800                   # the repaired recipe
bash scripts/dev/394_s2_proof/preflight.sh          # must print PREFLIGHT OK
bash scripts/dev/394_s2_proof/corridor_sampler.sh "$RUN/corridor_equal.csv" 1 &
ARM=equal   bash scripts/dev/394_s2_proof/boot_ab.sh
# bounded curl -m readiness loop, then the bench-length generations, then the
# teardown -- and only THEN the readout, on the final work-matched revision
python3 scripts/dev/394_s2_proof/read_arm.py <run> equal
ARM=compute bash scripts/dev/394_s2_proof/boot_ab.sh
python3 scripts/dev/394_s2_proof/read_arm.py <run> compute
# and the only place a cross-arm number comes from -- it refuses if the two
# arms did not do the same work:
python3 scripts/dev/394_s2_proof/read_arm.py <run> equal --against compute \
    --links 14.42,6.45,13.41
```

`run_arm.sh <arm>` does the whole sequence for one arm and is the driver to use;
the lines above are what it runs. It reads the dump twice — a pre-teardown
liveness check into `read_<arm>.txt` and the quotable post-teardown one into
`read_final_<arm>.txt`. Neither of those is a comparison: after both arms have
run, the window's number comes from the `--against` invocation, which reads both
final revisions itself and refuses rather than letting the work points be
checked by eye.

The `compute-cal` sub-arm is FALSIFIED and is not part of a measurement window
any more. To test a replacement hit-rate model against it, derive coefficients
from the EQUAL arm's own dump and pass the experimental symbol:

```
python3 -c "
from sglang.srt.layers.moe.expert_compute_placement import (
    cold_traffic_coefficients_from_measurement as c)
print(','.join(f'{x:.4f}' for x in c([B0,B1,B2],[0.485,0.42,0.42],[H0,H1,H2])))"
ARM=compute-cal SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS=<that> \
  bash scripts/dev/394_s2_proof/boot_ab.sh      # -> --rank-moe-ratio link-calibrated
```

`H0,H1,H2` are the equal arm's per-rank `h2d_bytes` and `B0,B1,B2` the base plan
THAT arm ran — read both off its own boot log, never assume them. Plain `link`
refuses while `SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS` is exported, so the two
arms cannot be confused for one another.

## Status

**Slice 3 has served tokens and the result is ACCEPTANCE-EVIDENCE. The
DESK-WRITTEN-NEVER-EXECUTED label is LIFTED** for the compute path as of the
2026-08-03 night window: the resolver reads the launch flag, residency is held
at the base plan on all three ranks, the arm harness carries the arm, and all
three 2026-08-02 defects are confirmed fixed ON HARDWARE. The green-corridor
re-proof then cleared the last open item.

The number to quote, work-matched on the final dump revision:

* **`compute` = 1.4307x** on the transfer term (green-corridor window,
  prediction 1.427x) and **-6.42 %** end-to-end against a 0.424 % same-window
  spread, in a corridor whose per-card minimum is 655-1318 MiB.
* The night window reads **1.4253x** on the same basis (prediction 1.411x). The
  two windows agree to 0.4 %.
* The pre-teardown readings — 1.5028x and 1.496x — are ~5 % high for the reason
  set out in "Which revision to read" and are not quotable.

What is still open:

* Nothing that owes a boot. The per-rank compute/wait split the night window
  skipped was read in the green window from the prefill clock: `equal` TP1 waits
  3.6 % of its time and `compute` TP1 rises to 15.3 % while TP2 becomes the new
  prefill clock at 5.0 % — the wait time moved rather than vanishing, which is
  the shape this file asked to see.

The three defects the 2026-08-02 battery found, and how each was closed:

| # | defect | fix | falsifier |
|---|---|---|---|
| 1 | the resolver read the resident fraction as `1.0` — it runs in the launcher, before the ServerArgs reach the runtime context, and the flag source swallowed the resulting error | the resolver hands its own `server_args` down to `resident_fraction_vector`; the env/flag cross-check is unchanged | `TestTheResolverReadsTheLaunchFLAG`, can-fail = restore the context-only route (5 tests fail) |
| 2 | the arm was not VRAM-neutral: residency was sized off the SOLVED expert count, +19.5 % on tp2 → OOM during staging. Correcting the fraction feeds back into the solve (a fixed point the spec did not address) | residency held at the pre-link base plan through a DERIVED sizing fraction on a channel the solve never reads — see the NOTE above | `TestResidencyIsHeldAtTheBasePlan`, can-fail = drop the correction (tp1 25≠31, tp2 37≠31) |
| 3 | the battery's driver set `ARM` without exporting it, so every arm booted the baseline; `boot_ab.sh` defaulted to `equal` rather than refusing | `run_arm.sh` is in the repo and exports; `boot_ab.sh` refuses an unset arm and has a `DRY_RUN=1` mode | `TestTheArmHarnessCarriesTheArm`, can-fail = restore the `equal` default (the unexported-arm case fails) |

All three were confirmed fixed on hardware by the confirmation window's Gates
1-4, which is what a can-fail arm cannot do on its own.

Hermetically tested: `test/registered/unit/layers/moe/test_expert_compute_placement_439.py`,
92 tests + 143 subtests, including an execution smoke of the full resolver path
with the hardware facts injected through `SGLANG_RANK_CARD_UUIDS` +
`SGLANG_MOE_HOST_SHARD_RATIO`, and eight proven can-fail arms (solve ignores the
links; resident mass allowed to float; launcher call removed; worker refusal
removed; a stale coefficient export silently recalibrating the solve; plus the
three above). The reserve infeasibility has its own hermetic reproduction in
`test/registered/unit/server_args/test_uneven_tp_args.py`
(`TestDerivedReserveInfeasibility`, 7 tests, can-fail = silence the note).

The catalog-first analysis behind the design is
`docs/dev/ANALYSE_439_expert_compute_placement.md`; the battery records are
`/spinning/gpu-battery-results/2026-08-02_439_arm3/RESULTS.md` (baseline only),
`/spinning/gpu-battery-results/2026-08-03_439_confirm/RESULTS.md` (the night
window, corridor-red) and
`/spinning/gpu-battery-results/2026-08-03_439_green/RESULTS.md` (the
green-corridor re-proof, and the source of the work-matched rule and of every
number this file now quotes as acceptance-evidence).
