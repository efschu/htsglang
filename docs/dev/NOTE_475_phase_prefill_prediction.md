# NOTE 475 — why the prefill-optimal layout did not win the prefill phase

The question this note answers, in the form it was asked: *if the
prefill-optimal layout and the decode-optimal layout do not each win their own
phase, something is wrong, and it is not explainable by hardware.*

The premise is right and the conclusion is right. On this rig the
prefill-optimal layout **does** win the prefill phase on the FP8 checkpoint,
by +15.2 % and +18.0 % of prefill-window time in two independent boots. On the
INT8-W8A8 checkpoint the same recipe measured −1.2 % and +1.8 % against a
predicted +5.8 % / +6.2 %. The defect is in the **cost model**, at one
nameable seam, and it is worth about 27 ms per 1000 prompt tokens.

Everything below is re-derived from logs that already existed. No boot was run
for this note.

---

## 1. The instrument, and what its two numbers mean

Every prefill batch logs, per rank,
`gpu-ms: T (compute Tc, wait Tw)` — `wait` is device time spent *inside*
collectives (`python/sglang/srt/utils/collective_clock.py`), `compute` is the
rest of the forward span, and `T = Tc + Tw` holds per rank while `T` is common
to all ranks (they run lockstep). Two quantities follow with no modelling:

```
max_r compute_r = T − min_r wait_r     the critical rank's own work
min_r wait_r                            what no weight re-split can remove
```

`scripts/dev/475_prefill_barrier/window_accounting.py` extracts them, grouping
batches into probe clusters by a >10 s idle gap and matching the clusters to
`raw/punkte.jsonl` in order. Full 2048-token chunks only; ragged tails are not
comparable across arms. Sample sizes are 43–52 batches per arm.

## 2. The measured prefill window, per layout

Per 1000 prompt tokens, s=1 probe, same rig, same day, barlink BAR1 except
where noted.

| arm | MLP vector | window T | critical-rank compute | collective |
|---|---|---:|---:|---:|
| `#424 fp8_decode` | VRAM-auto split | 803.3 | 290.7 (36.2 %) | 512.6 |
| `#424 fp8_prefill` | 10,1,1 | **697.2** | 165.5 (23.7 %) | 531.6 |
| `#435 fp8_decode_bar1` | VRAM-auto split | 744.2 | 331.6 (44.6 %) | 412.5 |
| `#435 fp8_prefill_bar1` | 10,1,1 | **630.8** | 185.0 (29.3 %) | 445.7 |
| `#424 int8_decode` | VRAM-auto split | 527.4 | 117.9 (22.3 %) | 409.5 |
| `#424 int8_prefill` | 10,1,1 | 533.8 | 83.4 (15.6 %) | 450.3 |
| `#433 int8_prefill_solved` | 8,1,1 (solved) | 518.1 | 80.7 (15.6 %) | 437.4 |
| `#435 int8_match_B2` | 8,1,1 + `--rank-kv-ratio capacity` | 545.4 | 109.7 (20.1 %) | 435.7 |

Provenance: `/spinning/gpu-battery-results/2026-08-02_424_phase_record_bench/raw/server_*.log`,
`…/2026-08-02_433_int8_prefill/raw/server_int8_prefill_solved.log`,
`…/2026-08-02_435_coupling_fp8bar1/raw/server_*.log`; line ranges are printed
by the extractor.

Read the INT8 rows first. **The concentration did exactly what it was asked
to do**: the critical rank's own compute fell 117.9 → 80.7 ms/1k, −31.6 %, and
the per-rank computes went from `[60.0, 112.6, 117.9]` to `[76.2, 77.6, 80.7]`
— from lopsided to balanced. The round got no shorter, because the collective
share rose 409.5 → 437.4, +27.9 ms/1k, which is 75 % of the compute saving,
and the rest is eaten by the mild over-prediction of the compute term itself.

The FP8 rows show the same collective growth (+19.0 and +33.2 ms/1k) against a
compute saving four times larger (−125.2 and −146.6 ms/1k), so there the
concentration pays and the layout wins its phase.

The one variable separating the two checkpoints is the rank spread of the GEMM
lane. The rig's own plan logs: INT8-W8A8 runs `681.4 / 187.6 / 183.8` TFLOPS
(both 3080s on the *native* int8 lane, 3.7:1) and FP8 runs
`563.1 / 57.6 / 60.8` (both 3080s on weight-only Marlin, 9.8:1). Under FP8 the
weak cards are weak by an order of magnitude and there is a great deal of
compute to move; under INT8 they are competent and there is not.

## 3. Where the collective growth comes from — and why the model priced it at 0

`PerfCostModel._prefill_sharded_time` computed

```
t = max_rank ( sum_family t[family][rank] )  +  t_all_reduce  +  invariant
```

Its own comment says the round contains "two all-reduces of H bf16 **per
layer** per token". Those two statements are inconsistent. If the group
synchronises after the attention block and again after the MLP block of every
layer, the lockstep max applies **per barrier**:

```
t_lockstep = sum_family ( max_rank t[family][rank] )
```

and by Jensen `sum max ≥ max sum`, with equality exactly when one rank is
slowest in *every* family. The old form was therefore a lower bound — tight on
a rig whose weak card is weak everywhere, loose precisely where
`--rank-perf-tune phase-prefill` operates, because concentrating the MLP onto
the strong rank is the operation that makes the MLP-slowest and the
attention-slowest rank different ranks.

The gap is now a first-class quantity, `PerfCostModel.prefill_barrier_skew`,
and it is reported per candidate in the plan log.

**The anchor.** For INT8 `8,1,1` against the VRAM-auto split, on the rig's own
probed rates and the checkpoint's own family param counts, the skew term is

```
27.6 ms per 1000 prompt tokens        (predicted, zero fitted parameters)
27.9 ms per 1000 prompt tokens        (measured collective growth, §2)
```

`#433` is the clean point for this comparison: against `#424 int8_decode` it
changed the MLP vector and nothing else — same checkpoint, same context, same
probes, same barlink BAR1 transport, and the same DCP token vector
`[31, 17, 16]` (that vector being wrong for the solve is #435's finding, not
this one; both boots carried the identical wrong vector, so it cancels).

Note where the time is charged in the instrument. A rank that arrives early at
a barrier waits *inside* the collective's CUDA-event span, so barrier skew is
booked as `wait`, not as `compute`. That is why the measured `compute` axis
follows `max_rank sum_family` (at a uniform 0.88 efficiency factor, base and
concentrated alike) while the *round* follows `sum_family max_rank`.

## 4. Backtest

`scripts/dev/475_prefill_barrier/backtest.py`, desk-only:

```
arm                                     pre-#475   shipped  measured  skew ms/1k
FP8  base -> 10,1,1  (#424, NCCL)         +18.3%    +18.3%    +15.2%        0.0
FP8  base -> 10,1,1  (#435, BAR1)         +18.3%    +18.3%    +18.0%        0.0
INT8 base -> 10,1,1  (#424)                +5.8%     +1.9%     -1.2%       27.0
INT8 base ->  8,1,1  (#433, solved)        +6.2%     +2.2%     +1.8%       27.0

rms error (points)                           4.4       2.2
```

The measured column is the prefill WINDOW, not the probe's host-side tok/s;
the two disagree by up to 9 points on the same arm pair (`#424` FP8: +24.1 %
probe against +15.2 % window), and the window is the quantity the cost model
predicts.

What changed and what did not:

* FP8 is **byte-identical**. The 3080s pace every barrier at every candidate
  vector, so the skew is exactly 0 and the two forms return the same
  float. The measured +15.2 / +18.0 % keeps its +18.3 % prediction, and the
  #216/#230 calibration anchor (+6.4 / +9.0 / +13.0 % measured for 3,1,1 /
  4,1,1 / 6,1,1) is unmoved.
* INT8 drops from a reportable claim to a claim below the floor. Both boots
  measured their own A-vs-A prefill floor at 3.0 % and 3.5 %; the shipped
  model now predicts +1.9 % and +2.2 %, i.e. "not resolvable by this
  instrument", which is what four measurements said.
* The INT8 argmax moves 8,1,1 → 4,1,1. That is the model preferring a milder
  concentration because over-concentrating buys balanced compute at the price
  of a barrier the round then pays twice. The whole INT8 ladder is +3.0 % to
  +4.6 %, i.e. inside the floor — the honest summary is *INT8 has no prefill
  lever on this rig*, and the planner now says so.

Residual honesty: the shipped model is still optimistic by ~3 points on the
INT8 arms (+1.9 predicted against −1.2 measured on 10,1,1). The barrier term
accounts for 27 of the 41 ms/1k the `#424` INT8 pair moved; the remaining
14 ms/1k is not explained here. That arm also changed `--rank-kv-ratio` to
`2,11,10`, which redistributes DCP token ownership and therefore prefill
attention work — a second skew source the family model does not represent at
all. `#433`, which changed only the MLP vector, has no such residual.

## 5. Two stale fixtures the fix exposed

Both had been masking the defect by feeding the model rates the hardware does
not run at, and both were found by this change rather than caused by it.

**`test_prefill_calibration._GEMM = [233.91, 63.17, 61.24]`** (written
2026-07-27) is the generic stage-0 lane. #298a (2026-07-30, "score the prefill
objective in the checkpoint's own GEMM format") replaced it three days later;
the same rig now prints `563.1 / 57.6 / 60.8` for the same FP8 checkpoint. The
hardware never changed — the 3080s always went through Marlin — only the
probe's reading of it, from a 3.7:1 spread to the real 9.8:1. With the stale
vector the skew at 6,1,1 is a spurious 67.5 µs/token; with the real one it is
0 and the measured +13.0 % is reproduced. Fixture refreshed with provenance.

**`key_solver.check_regressions` scored prefill on the PRE-resolution rates.**
It built the cost model from `rates.resolve_gemm_format(...)` and then read
`rates.require_gemm_tflops()` off the unresolved object, so both FP8 anchors
were priced on the dense bf16 fallback. Fixed to read `model.rates`; the
function also grew a `hardware_profile` passthrough so an anchor can be scored
on the lane the measured boot actually dispatched to. Both anchors pass under
both forms of the arithmetic after the fix (`264_611_net_negative`: measured +8.2 %,
predicted +9.97 % without the lane profile, +10.62 % with it, tolerance 4.0).

## 6. The floor explosion is a separate, smaller defect — in the harness

`#435` sub-arm B2 reported a 13.0 % A-vs-A prefill floor from three identical
draws (1597.7 / 1720.2 / 1820.2 tok/s) where `#424` reported 3.0 %. That 4.4×
is not admission noise and not the `capacity` KV vector. From the same
CollectiveClock lines, per 1000 tokens:

| draw | window T | critical-rank compute | collective |
|---|---:|---:|---:|
| `int8_match_B2_floorP1` | 591.5 | 226.0 | 365.5 |
| `int8_match_B2_floorP2` | 538.6 | 171.9 | 366.7 |
| `int8_match_B2_floorP3` | 507.1 | 146.2 | 360.9 |

**The collective axis is flat to 1.6 %; the entire spread is compute, and it
is monotone.** The same shape appears in `#435 int8_match_B` (255.4 → 182.1 →
145.9, collective 367.6 / 368.4 / 360.1) and, weaker, in `#433` (169.9 →
119.7 → 106.9, collective flat within 0.9 %). The draws are 48–51 s apart with
~12 s of work each, so the cards spend ~75 % of each cycle idle and every draw
pays part of a clock ramp; the arms where the steady state is reached fastest
(`#424 int8_decode`: 120.6 → 113.3 → 113.4, non-monotone, 3.0 %) are the ones
that reported a tight floor.

Consequence for how the number is used: three draws with a monotone trend are
not exchangeable samples, so their spread is not a noise floor — it is a
systematic drift being reported as noise. It is loose in the direction that
lets a real regression pass as "within floor", which is exactly what happened
to `#435` sub-arm B (`−8.5 %` scored as parity against a 13.0 % floor). The
remedy is a warm-up draw that is discarded, or back-to-back draws with no idle
gap, not a wider floor. Not fixed here — it is harness work
(`scripts/gpu_battery/s12_prefill_kurve.py`), and it is registered as the
ticket in §8.

## 7. What shipped

* `uneven_perf.PerfCostModel.per_family_prefill_compute_times` — the per-family
  per-rank table the barrier max is taken over.
* `uneven_perf.PerfCostModel.prefill_lockstep_compute_time` — `sum_family
  max_rank`; `_prefill_sharded_time` now calls it.
* `uneven_perf.PerfCostModel.prefill_barrier_skew` — the Jensen gap, reported
  per candidate in the plan log as a share of the base step.
* `key_solver.check_regressions` — scores on the resolved rates, accepts a
  `hardware_profile`.
* `test/registered/unit/planner/test_prefill_barrier_skew_475.py` — 6 tests /
  29 subtests, including the 27.6-vs-27.9 anchor, the FP8 no-op guard, a
  symmetric-rig generality guard, and the Jensen non-negativity invariant.
  Can-fail arm: 31 of 35 fail on the pre-fix tree.
* `scripts/dev/475_prefill_barrier/{window_accounting,backtest}.py`.

Full planner suite: 2170 passed, 1 skipped, 249 subtests.

## 8. GPU confirmation ticket (not run here)

One boot pair settles the residual of §4 and the harness question of §6.

* Checkpoint `Qwen3.6-27B-INT8-W8A8`, tp 3 on 5090 + 2× 3080, ctx 131072,
  kv `fp8_e4m3`, barlink BAR1, NEXTN 3, decode graphs `full`, device order via
  `registry.nvml --map`.
* Arm A: `--rank-tp-ratio auto-performance --rank-perf-tune phase-prefill
  --rank-auto-reserve-mib auto` and nothing else. Expected: the solve now
  chooses a milder vector than `8,1,1` and the plan log states its barrier
  skew. Record the `CHOSEN` line and the skew.
* Arm B: the same command with `--rank-perf-tune phase-decode` (the base
  split), same boot session ordering as `#424`.
* Both arms: `--rank-kv-ratio coupled` — do NOT vary the KV vector, it is the
  unmodelled second skew source and it is what makes the `#424` INT8 pair
  irreducible.
* Harness change under test, and the reason the ticket is worth a boot: run
  the three floor draws **back to back with no idle gap**, plus one discarded
  warm-up draw before them. Predicted outcome from §6: the floor collapses
  from 13 % to the 3 % range, and with a 3 % floor the INT8 prefill/decode
  comparison becomes resolvable for the first time.
* Pass criterion for the cost model: the predicted prefill delta of arm A
  against arm B lands inside the (now tight) measured band, and the plan log's
  reported barrier skew matches the measured collective growth of the pair to
  within 20 %.
* Corridor: the usual ≥400 MiB free on every card; `#435` sub-arm B needed
  `--rank-auto-reserve-mib 4760,4160,4160` to hold it under a matched KV
  vector, and this ticket does not use one, so `auto` should suffice — verify
  from the 2 s sampler, do not assume.
