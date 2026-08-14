# TICKET #363 — ACT WINDOW VERDICT R3 (R15)

Window `363-act-r3`, 2026-08-14, held 17:59:15Z–19:33:24Z. Worktree
`/spinning/wt-merge-r15` (branch `merge/r15-batch`, frozen base `e61b0d4708` =
`origin/feat/route-a-631` = `origin/integration/r2`),
`PYTHONPATH=/spinning/wt-merge-r15/python`. Evidence
`/spinning/evidence-363-act-r3/`. Runsheet `docs/dev/363/RUNSHEET_363_ACT_WINDOW.md`.

**The headline.** R14 handed over defect 8 as a device-timing problem and named
the fix as "a decision, not a bug fix". It is neither. **R14's own B3 trace
falsifies its diagnosis**, and the real defect is a one-line mode gate that
made the measurement this ticket needs impossible to take on any rig. It is
fixed, and with it the chain reaches the end: the measurement pass has written
a canon, the stage table BUILDS, and the decision rule runs per boundary and
records its verdicts on real measured rounds.

---

## 1. DEFECT 8 WAS MISDIAGNOSED, AND THE EVIDENCE WAS ALREADY ON DISK

R14 recorded defect 8 as:

> `rank_compute_ms` / `rank_wait_ms` are "**None on a graph-covered forward**".
> On this rig every forward is graph-covered, so `_ms_split_n` never leaves 0.

Both halves are false, and B3's own artifacts say so.

**This rig does not graph-cover prefill.** B3's boot log, three ranks:

```
Disable prefill CUDA graph because cuda_graph_config resolved
prefill.backend='disabled'
```

`cuda_graph_config` in that same log reads `decode=PhaseConfig(backend='full')`,
`prefill=PhaseConfig(backend='disabled')`. Decode is captured; prefill runs
eager.

**The split was being measured, in the very boot that concluded it was not.**
B3 logged **2574** prefill lines carrying both terms and the family
decomposition:

```
Prefill rank batch, #new-token: 511, #cached-token: 0, #chunks: 1,
gpu-ms: 330.8 (compute 65.7, wait 265.1)
(wait by family: tp.all_reduce 248.9/129x, dcp.all_gather 14.8/16x, ...)
```

And the split REACHED the observer: `rank_mean_forward_ms` is non-null on
**82 341 of 82 549** verdict rows, on all three ranks, gated by the same
`last_split_known` flag that gates the compute/wait pair.

### 1.1 What was actually missing: the CLOCK

`build_regime_observer`:

```python
# Built only in act mode: in observe there is nothing for an admission gate
# to admit, and a clock whose verdict cannot move anything would be an
# expensive observe under a misleading name.
if mode == MODE_ACT and bool(getattr(server_args, "regime_stage_clock", False)):
```

B1/B2/B3 are OBSERVE boots (RUNSHEET §4.2). So `_stage_clock` was `None`,
`_intra_phase_decide` was never called, and `ms_decision` was `None` on every
row — not because the split was absent but because nothing asked for it. The
trace says so directly: `"stage_clock": null` in the summary line of **all
three ranks**, which is exactly `self._stage_clock is None`.

**This is a bootstrap deadlock one layer above the one R14 fixed as defect 7:**

```
stage_measure_pass reads OBSERVE traces to build the canon
  -> the canon is what makes a stage a flip target
    -> a flip target is what act mode requires
      -> but ms_decision rows are written only when the clock is wired
        -> and the clock was wired only in act mode.
```

R14's defect-7 fix is the "measurement only: no boot stage table" branch of
`_intra_phase_decide`, written for precisely this situation. It hung off a
clock that observe never constructed, so it could not execute. The canon could
not be bootstrapped on ANY rig, which is why four consecutive windows found a
different-looking wall.

The reasoning in that comment is right about the ADMISSION GATE and wrong about
the CLOCK. A gate prices a flip against the corridor; observe has no flip to
price, and a gate is a step toward acting. A clock is a way of SEEING. The
clock now follows the flag in every mode; the gate stays act-only, so observe
gained an instrument and still holds no actuator path — the no-actuator import
pin is untouched and still green.

---

## 2. DEFECT 8b — A MEAN OVER THIRTY COPIES OF EACH SAMPLE

Found while proving 8a, and it had to be fixed in the same window or the first
measurement this ticket ever took would have been a false one.

`RankPrefillLog` carries its last measurable reading forward until another one
flushes. That is right for a LOG LINE ("what was last seen") and wrong for a
MEAN: the accessor answers every later boundary with the same retired forward,
and the observer accumulated it once per BOUNDARY. The arithmetic on R14's B3:

| | count |
|---|---|
| boundaries carrying a number | **82 341** |
| prefill forwards actually measured in the whole boot | **2 574** |

A mean over roughly **thirty copies** of each real sample. It reads exactly
like a measurement and is not one, and it is invisible in every summary
statistic — which is why it survived four windows.

The split now carries a monotone `last_split_seq`, advanced only on a flush
that sets a split, and a boundary accumulates only a reading it has not already
counted. The hook's tier pin was updated deliberately, with the tier argument
recorded in the test's own docstring.

**This is also why R14's A-vs-A floor of 0.03 % was not a floor.** Averaging
thirty copies of each of a handful of samples makes two arms agree to three
decimals no matter what the rig is doing. §5.1 has the honest number.

---

## 3. DEFECT 5 — RESOLVED STATE, NOT THE FLAG

`_booted_stage` read `server_args.rank_mlp_ratio`. Under `--rank-tp-ratio
auto-performance` that flag is `None` while the server runs a concrete resolved
partition; `reachability()` checks weights FIRST; so every candidate came back
`REACH_NO_WEIGHT_MOVER` whatever its KV vector.

It now reads `get_tp_partition_ratios("mlp")` — what the layers themselves
partition on, which answers with the installed `mlp` family vector and falls
back to the base plan exactly as the layers resolve it. The flag survives only
as a last fallback for a caller holding `server_args` before the plan is
installed.

**Canonicalisation, and the space this deliberately does not reach.** A
candidate reports `key_solver._ratio_of(units)`, a gcd-reduced ratio of MLP
UNITS; the installed plan is a raw ratio. Both sides are now gcd-reduced into
one space, so `[60, 34, 42]` and `[30, 17, 21]` compare equal — they describe
one partition. The EXACT test is in units, because `partition_units` is a
largest-remainder quantisation and is many-to-one: two ratios that reduce
differently can still land on the same units. Reducing is therefore **stricter**
than the exact test — it can refuse a stage that is in fact reachable, and can
never admit one that is not. That is the safe direction and it is deliberate;
the exact test needs the checkpoint's unit count, and making the stage table
depend on a built cost model to answer a reachability question is a much larger
coupling than the residual it removes. Named in the code at the point of
comparison.

---

## 4. THE ROUTE R14 PROPOSED IS CLOSED ON THIS STACK

R14 (§9.6) and the shift brief both proposed instrumenting inside the graph
replay: record CUDA events around the replay and inside it, read them with the
staggered pattern. The first half already exists (`DeviceTimer.wrap()` brackets
the decode replay). **The second half is impossible on this stack, measured
rather than assumed** — `evidence-363-act-r3/probe_graph_events*.py`:

| probe | result |
|---|---|
| record `torch.cuda.Event` into a capture | **succeeds** — the capture accepts the event-record node |
| `Event.query()` after replay | `cudaErrorInvalidValue` |
| `Event.elapsed_time()` after replay, never having called `query()` | `cudaErrorInvalidValue` |

The third row is the one that settles it: a CUDA error is sticky within a
context, so probe 2 could not distinguish "elapsed_time is unsupported" from
"poisoned by the query that preceded it". Probe 3 never queries a
graph-recorded event — readiness comes from an ordinary OUTER event recorded
around the replay call — and `elapsed_time` still fails. torch 2.11.0+cu13.

So the wait term inside a FULL captured graph is not readable here, and a
future shift should not spend a window rediscovering that. It did not matter:
the terms this ticket needs come from the eager prefill path, which was
measuring them all along.

**Instrument overhead: zero device work added.** No CUDA event, stream
operation or synchronisation was added by any fix in this window. 8a
constructs one Python object at boot. 8b adds one integer compare per
consensus boundary (one round in eight) and one integer increment per prefill
flush. The device-side cost of the split remains the #252 CollectiveClock's
own, measured at 0.13 % on the path that was already paying it.

---

## 5. THE METAL RUN

Four boots, port 30041, all carrying `--tp-size 3 --rank-gpu-id 0,1,2
--rank-tp-ratio auto-performance --rank-mlp-ratio 94,13,29 --rank-kv-ratio
30,17,17 --rank-auto-reserve-mib 5500,3800,3800 --kv-cache-dtype fp8_e4m3
--context-length 32768 --max-running-requests 16 --speculative-algorithm NEXTN
... --kv-reshard-vectors '30,17,17;1,1,1' --regime-stage-clock`. Driver
`--repeats 6 --burst 16 --burst-tokens 6000 --drain 12 --drain-tokens 900
--mixed 8 --idle-s 25` on B1/B2/B3, `--repeats 2` on B4.

| boot | role | requests | failed |
|---|---|---|---|
| B1 | floor A | 216 | 0 |
| B2 | floor B | 216 | 0 |
| B3 | reference + stage segments + flip samples | 216 + 216 | 0 |
| B4 | the rule with the canon present | 72 | 0 |

The line that could not exist before, on all three ranks of every boot:

```
REGIME-OBSERVE intra-phase axis MEASURING (#363): the ms/round clock is wired
in observe mode and writes ms_decision rows, but holds no admission gate and
no actuator. These rows are what stage_measure_pass turns into the canon.
```

### 5.1 The split is real, and the freshness fix is visible in it

B1, ms_decision rows cross-checked across the three rank files:

| property | result |
|---|---|
| rounds where all three ranks report the IDENTICAL split | **12 of 12** |
| distinct `mean_total_ms` values among those 12 | **12 of 12** |
| `mean_wait_share` range | 0.190 → 0.621 |

The first row is the group reduction working: the clock is fed the
MIN-reduced statistic, so its inputs are replicated and its verdict is uniform
by construction. The second is defect 8b fixed — **every sample is a different
retired forward**; under the old behaviour this column would have been a short
list of values each repeated many times. The third is the signal the axis
exists to see, moving with the load shape rather than sitting flat.

Coverage where it matters: in `prefill_heavy`, **every** boundary carried a
split (B1 5/5 early, 53 of 71 over the full run once the idle stretches are
included), because during a prefill burst a forward retires every boundary.

### 5.2 The A-vs-A floor, honestly, for the first time

| | value |
|---|---|
| floor, warmup 5 (used by the pass) | **4.43 %** |
| floor, warmup 20 | **9.09 %** (B1 1068.774 ms vs B2 1175.589 ms) |
| R14's floor, same driver | 0.03 % |
| R13's floor | 1.48 % |

**R14's 0.03 % was an artifact of defect 8b**, not a property of the rig: a
mean over ~30 copies of each sample is reproducible almost by construction.
The honest floor on this rig, from samples counted once, is 4.43 % at the
warmup the pass used. It is **below** neither the shipped
`DEFAULT_ENTER_MARGIN_PCT` of 5.0 (at warmup 5) nor above it by much, and at
warmup 20 it exceeds it. The watermark was NOT moved: per TICKET P2 it moves
once, before a window, recorded with its measurement — never afterwards to
accommodate a result.

**Warmup 5, applied uniformly, and why.** At warmup 20 the reference segment
carried 2 boundaries and the pass refused it (`below the 8 this pass
requires`) — correctly. Warmup 5 was then applied to reference, stage AND
floor together, so the instrument stays internally consistent; the reference
arm then carries 17 boundaries and the stage arm 51. This is recorded because
a warmup chosen after seeing a refusal is exactly the kind of choice that has
to be stated rather than buried.

### 5.3 The reshard guard refused again, and the server stayed up

R14's defect-2 guard fired collectively on the first attempt under load, round
**57512**, and this is an INDEPENDENT reproduction of its central claim:

| rank | free | corridor floor | transient needed | margin |
|---|---|---|---|---|
| 0 | 2389.7 MiB | 1024.0 | 3557.0 (staged 1159.9 + packed 1159.9 + pack-peak 580.0 + recv 657.2) | **−2191.3** |
| 1 | 4538.4 MiB | 1024.0 | 2554.4 (658.3 + 658.3 + 329.4 + 908.3) | **+960.0** |
| 2 | 3680.4 MiB | 1024.0 | 2551.0 (656.7 + 656.7 + 328.3 + 909.3) | **+105.4** |

Group MIN −2191.3 → REFUSED, **two of the three ranks could afford the move**
and would have allocated under a per-rank guard. Server up, corridor untouched,
request stayed on the incumbent layout.

`POST /flush_cache` REFUSES under load ("When there are running or waiting
requests, the operation will not be performed"), so the drained-then-flush
sequence is not available mid-driver; the cutover was taken after the driver
finished. It then committed in **4.0–4.1 ms**, `0 live slots`. Seven flips over
the boot (21 `DONE` lines, three ranks each), **1.2–4.1 ms**, every one an
EMPTY-POOL LOWER BOUND. RUNSHEET §4.4's warning stands in full, and R14's item
4 is still open.

### 5.4 The measurement pass wrote a canon — the first one

```
planner:maxkv [prefill_heavy] vs booted: gain -3.16 % band 4.43 %
flip 0.00 s (21 flip sample(s), 564/182 s covered) -- REFUSED (1)
  REFUSED: gain -3.16 % does not clear its own band of 4.43 % (#360):
  a difference inside the band is not a difference, however it is labelled
```

Record: `gain_pct -3.157609`, `band_pct 4.427616`, `flip_cost_s 0.0041`
(MAXIMUM of 21 samples, per the runsheet), `flip_cost_mean_s 0.00209`,
`boundaries_reference 17`, `boundaries_stage 51`, `covered_s 182.1 / 563.6`,
`drift_pct 3.204`, rig key resolved from NVML UUIDs.

Both arms clear the 10 s device-time floor with large margin. The verdict is
that **on this rig, at this operating point, the `maxkv` layout is not
distinguishable from the booted one** — its gain is negative and inside the
band either way.

### 5.5 The stage table BUILDS, and the refusal is now a measured one

B4, with the canon on disk:

```
REGIME-OBSERVE stage measurement canon: 1 record(s) ... rig 3:GPU-31d7ef41...
REGIME-OBSERVE stage measurement: planner:maxkv: NOT SELECTABLE --
  stage 'planner:maxkv' HAS a measurement and it is refused:
  gain -3.16 % does not clear its own band of 4.43 % (#360)
REGIME-OBSERVE stage table: 1 stage(s), 1 reachable at runtime,
  0 flip target(s), booted on 'booted'
```

**The table builds.** For four windows it did not exist at all
(`could not build the boot stage table`). The refusal has moved through its
whole ladder across three windows:

| window | why there was no flip target |
|---|---|
| R13 and earlier | the planner could not solve — `PlannerFeedUnavailable` |
| R14 | the planner solved 1 stage but it **carries no measurement** |
| **R15** | the stage **HAS a measurement and the measurement says no** |

### 5.6 The decision rule ran, and recorded its verdicts

With the table present, `_intra_phase_decide` reaches
`self._stage_clock.decide(...)` every boundary. B4, 29 `ms_decision` rows,
`mean_total_ms` 310.9–1388.5 ms from real measured rounds:

| rows | verdict |
|---|---|
| 3 | `ms window not ready: 1 / 2 / 3 of 8 samples. A flip decided on fewer samples is decided by those samples.` |
| 22 | `no measured candidate differs from the stage in force; the ms axis has nothing to compare` |

**`signal_pct` is `null` on these rows, and that is the rule reaching a
verdict rather than the machinery failing.** `_score` skips a candidate the
canon refused, so with one solved stage and that stage not selectable, the
candidate set is empty and there is nothing to compute a signal against. The
arithmetic the rule would apply is unchanged and pinned by the hermetic
suite: `signal = 100·(total − predicted)/total`, `band = √(band_i² + band_c²)`,
`flip_cost_pct = 100·flip_cost_s / payback_s`,
`threshold = max(enter_margin, band + flip_cost)`, adopt on
`signal > threshold` sustained over the enter window.

For the candidate this rig actually has, that arithmetic is:
**signal −3.16 % against a threshold of max(5.0, 4.43 + 100·0.0041/payback) —
a refusal by 7.6 points before the flip cost is even added.** No flip was
taken, and none should have been.

### 5.7 The six criteria

| ID | criterion | result |
|---|---|---|
| A1 | `stage_clock_proposals > 0` and `actuations > 0` | **FAIL** — 0 and 0, and now for a *measured* reason (§5.5) |
| A2 | flips over the window ≤ 4 | PASS |
| A3 | ms/round in SHIFT beats the control in the wait term | **ANSWERED, NEGATIVE** — −3.16 % inside a 4.43 % band |
| A4 | zero corridor samples below 1024 MiB, every boot | **PASS — all four boots** |
| A5 | `desyncs == 0`, summary present | **PASS** — summaries on all 3 ranks of all 4 boots |
| A6 | every `ms_decision` carries the rule | **PASS** — 29 rows in B4, each with its reason and measurement fields |

A3 and A6 move from UNANSWERABLE/VACUOUS to answered. A1 remains FAIL, and the
distinction that matters is that it is no longer unexplained: the rig's only
solved stage was measured and is not better.

---

## 6. CORRIDOR

Instrument's can-fail arm first: `corridor_report.py --smoke` **3/3** (clean
PASSES, planted 900 MiB sample FAILS, all-999 series FAILS).

100 ms sampling, NVML FREE, every boot:

| boot | samples/card | gpu0 min | gpu1 min | gpu2 min | verdict |
|---|---|---|---|---|---|
| B1 | 10 329 | 4471 | 2294 | 3627 | PASS |
| B2 | 10 311 | 4471 | 2294 | 3627 | PASS |
| **B3** | **24 090** | **4471** | **2294** | **3625** | **PASS** |
| B4 | 3 888 | 4541 | 2392 | 3683 | PASS |

**48 618 samples per card, ZERO below 1024 MiB.** B3 is the one that matters:
it contains two refused reshards and seven admitted ones.

### 6.1 `kv_ascend_mark`, recorded with its arm

| boot | peak occupancy | vs 0.85 |
|---|---|---|
| B1 | 0.651827 | UNREACHED |
| B2 | 0.651949 | UNREACHED |
| B3 | 0.651848 | UNREACHED |
| B4 | 0.651905 | UNREACHED |

Arm: `--rank-mlp-ratio 94,13,29 --rank-kv-ratio 30,17,17`, driver
burst16/6000. This reproduces R14's 0.6518/0.6516/0.6519 to four decimals on
an independent set of boots and confirms its §10.1 finding: gate 3's
reachability is a property of the ARM (this one funds a larger KV pool), not
of the rig. R14's item 3 is discharged — the reading is recorded with its arm.

---

## 7. TEARDOWN

Corridor sampler stopped and each `corridor.csv` closed before its boot.
Summary line present on all three ranks of all four boots, verified by count.

Heartbeat stopped **before** the holder was released (`.hb-stop-363-act-r3`,
unit `hb-363-act-r3.scope` confirmed inactive, last beat 19:26:35Z), aged to
**133 s** before the restore so the restore script's own 120 s peer-heartbeat
guard passed on the first attempt.

Serving on 30030 restored via the sanctioned `res-r5 restore_ship.sh` in scope
`ship-restore-r15.scope`, verified with a **REAL GENERATION** — text exactly
`MERIDIAN43`, `finish_reason=stop`, `completion_tokens=6`.

**One honesty note on the verify.** The first attempt returned empty text with
`finish_reason=length` and `completion_tokens=16`. That is NOT the #622 wedge:
it is the model's thinking mode consuming a 16-token budget. Re-run with
`enable_thinking: false` — this rig's standing default — and the generation is
clean. A verify that does not send thinking-off can manufacture a
wedge-shaped result out of a healthy server, which is worth knowing before
someone reads one as an incident.

Raw: `/spinning/evidence-363-act-r3/restore_verify.json`. Port 30099 never
touched. No broad `pkill`; every process stopped by its own unit name.

---

## 8. WHAT THE NEXT SHIFT OWES, IN ORDER

1. **A candidate worth flipping to.** The machinery is now end to end
   functional and the rig's only solved stage is measurably not better. The
   ticket needs either a second declared vector whose solve differs more from
   the booted one, or a regime where `maxkv` actually wins. `decode_heavy` and
   `mixed` both had too few boundaries this window (< 8) to measure; a driver
   shaped to produce them would answer it.
2. **The flip cost is STILL a lower bound.** All 21 samples ran on an empty
   pool, because the guard correctly refuses a loaded one at this operating
   point and `flush_cache` refuses under load. A real flip cost needs more
   headroom or a smaller redistribution than `30,17,17 -> 1,1,1`.
3. **The stale carry in `rank_forward_ms_from`.** Defect 8b was fixed for the
   #363 SPLIT accessor only. The older `rank_forward_ms_from` still carries
   its last reading forward, and it feeds the existing consensus/spread
   machinery, which has pinned tests and metal history. It is very likely the
   same class of bias in `rank_ms_spread_pct`; it was left alone deliberately
   rather than changed under a window's time pressure.
4. **The act entry gate.** Three of its four items (`f2_live_replay`,
   `f3_bands_measured`, `f4_card_comparison`) have no evidence on this rig, so
   act mode cannot be booted honestly. §5.6 shows the rule does not need act
   mode to be exercised — only to actuate — so this is no longer on the
   critical path for measuring, but it is on the critical path for flipping.
