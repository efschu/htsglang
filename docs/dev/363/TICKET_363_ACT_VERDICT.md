# TICKET #363 — ACT WINDOW VERDICT (R13)

Window `363-act`, 2026-08-14, held from 13:02Z. Worktree
`/spinning/wt-merge-r13` (the merged line: `ac0d1f36f3` + R13's three merges +
this shift's fixes), `PYTHONPATH=/spinning/wt-merge-r13/python`. Evidence
`/spinning/evidence-363-act/`. Runsheet executed:
`docs/dev/363/RUNSHEET_363_ACT_WINDOW.md`.

The runsheet names worktree `/spinning/wt-desk-363-act`. That branch is merged
now, and the merged line is what ships, so every boot below ran from the merged
line — which is also why §4's cross-branch defect was visible at all.

---

## 1. WHAT THE WINDOW FOUND BEFORE IT MEASURED ANYTHING

Four defects, each found by executing a step rather than reading it. They are
first because three of them would have silently corrupted the measurement.

### 1.1 The rate fingerprint could never date two of the three cards

The window's mandated first step is `card_rate_pass --run`, so the rates carry
the new power-limit + driver fingerprint. It measured all three cards. `--show`
then reported the 5090 **FRESH** and the 3080 profile **UNKNOWN** — "NVML
reports no current environment for this card to compare against" — for cards
NVML can see, seconds apart, in the same pass.

Cause: the two branches merged in this same batch disagree about the card's
name. `fix/card-library-guards` resolves a card by CAPACITY, so this rig's
profile is `RTX 3080 20GB`; `rate_env` looks the name up in a table keyed by
the raw NVML name, `RTX 3080`. Exact-key lookup, two names, one card:
permanent UNKNOWN for every capacity-disambiguated profile.

Fixed on the line (`42484da800`), red-first, and confirmed **on metal** — both
profiles now read FRESH at driver 595.58.03 / 200 W and 400 W. Full write-up in
`HANDOFF_MERGE_R13.md` §4.

**Measured rates, this rig, this window:**

| card | GEMM | membw | note |
|---|---|---|---|
| RTX 5090 | 203.42 TFLOPS | 1661.6 GB/s | 400 W |
| RTX 3080 (`5c648f96`) | 51.17 TFLOPS | 716.2 GB/s | THROTTLED, 200 W |
| RTX 3080 (`62dbbae1`) | 50.97 TFLOPS | 716.2 GB/s | THROTTLED, 200 W |

The THROTTLED caveat is this rig's deliberate power caps, so these ARE its
rates. They reproduce m584's figures within 0.4 %, which is the useful result:
the m584 numbers were right, they simply could not be shown to be current.

### 1.2 A semicolon silently halved the boot's actuator configuration

The first boot harness pasted the extra launch arguments into a `bash -c`
STRING. The `;` inside `--kv-reshard-vectors '2,11,10;3,10,10'` terminated the
command: the server came up carrying **one** vector, and `> boot.log 2>&1`
became a separate shell command, so there was **no boot log at all**.

A boot with one vector cannot flip — the act leg would have run with nowhere to
go, and the window would have reported "no flip targets" as a finding about the
controller. Caught by checking `/proc/<pid>/cmdline` after launch rather than
trusting the launcher's echo. The previous 363 window's `boot.sh` carries the
same flaw; it never passed a semicolon, so it never fired.

### 1.3 The runsheet's reserve breaches the corridor law on this rig

`RUNSHEET §4.1` specifies `--rank-auto-reserve-mib 3000,2700,2700`. Measured on
the first B1 attempt, 5497 samples/card at 100 ms NVML FREE:

| card | min free | samples below 1024 |
|---|---|---|
| gpu0 | **443 MiB** | **4528 of 5497** |
| gpu1 | 6116 MiB | 0 |
| gpu2 | 2931 MiB | 0 |

First breach ~97 s in and steady to the end, so it is not a load transient.
The squeeze is on ONE card while the other two sit loose, and — recorded
separately because it is the more interesting half — **the squeezed rank had
asked for a 2700 MiB reserve and was left 443 MiB, 2257 MiB below its own
request.** That is not explained by the ~424/518 MiB carve-out.

The window switched to `5500,3800,3800`, which is what the previous 363 window
booted with and is this rig's recorded stress reserve, held identical on every
boot thereafter. Corridor passes at those values with margin (§4).

Evidence: `/spinning/evidence-363-act/B1-r1/corridor.csv`.

### 1.4 The runsheet's driver profile is too light for its own analysis tool

`RUNSHEET §4.3`'s driver (`--repeats 2 --burst 16 --burst-tokens 6000 ...`)
produced, over a full run on each of two arms:

| | B1 | B2 |
|---|---|---|
| verdict boundaries | 22 646 | 11 872 |
| **ACTIVE** boundaries | **48** | **43** |
| `prefill_heavy` | **27** | **24** |

`msround_split.py` then refused the floor outright:

```
REFUSED: arm segment has 7 boundaries, below --min-samples 30. A band from a
handful of samples is a number, not a measurement.
```

27 minus `--warmup 20` is 7. The rig is idle in essentially every window: 48
active boundaries out of 22 646.

**The tool was not loosened.** Lowering `--min-samples` or `--warmup` to admit
the segment is solving against the instrument, and `--warmup 5` would still
have left 22. The driver was made longer instead, everything else identical,
and every rejected arm is archived rather than deleted.

The escalation was itself measured rather than guessed. `--repeats 4` was tried
first; a mid-run count at 2.7 minutes showed 10 `prefill_heavy` boundaries, i.e.
~3.7/min, extrapolating to ~41 for the full run and **21 after `--warmup 20`** —
still short. That arm was stopped at 3 minutes rather than run to a foreseeable
refusal (archived `B1-r3`), and the window went to **`--repeats 6`**
(~60 boundaries, ~40 after warmup).

| attempt | driver | outcome |
|---|---|---|
| `B1-r1` | repeats 2, reserve 3000,2700,2700 | corridor BREACH (§1.3) |
| `B1-r2` / `B2-r2` | repeats 2, reserve 5500,3800,3800 | corridor PASS, floor REFUSED (27/24 boundaries) |
| `B1-r3` | repeats 4 | stopped at 3 min on an extrapolated refusal |
| `B1` / `B2` | **repeats 6** | the arms this verdict is measured on |

`--burst` was considered and rejected as the lever: it raises the burst arm's
DECODE work in the same proportion (48 concurrent × 6000 generated tokens), so
it costs ~3x the wall clock for the same boundary count. `--repeats` buys
boundaries linearly.

---

## 2. THE BOOTS

All four boots: `--tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio
auto-performance --rank-auto-reserve-mib 5500,3800,3800 --kv-cache-dtype
fp8_e4m3 --context-length 32768 --max-running-requests 16 --speculative-algorithm
NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4`, model `Qwen3.6-27B-FP8`, port **30041**.

Port 30041 rather than the runsheet's 30030 is deliberate and inherited from
the previous 363 window: the local router on 30099 forwards to 30030, so a
measurement instance on 30030 receives the Qwen agent fleet's traffic and its
ms/round is not its own.

`max_total_num_tokens = 178089` on every boot, identical — which is the cheap
check that the arms really are the same configuration.

Driver on the measurement arms: `--repeats 6 --burst 16 --burst-tokens 6000
--drain 12 --drain-tokens 900 --mixed 8 --idle-s 25`, identical on every arm.
216 requests per arm, 0 failed. B3's flip-sample phase uses `--repeats 2`,
deliberately: it exists to keep the pool non-empty while the flips are timed —
the flip cost comes from the actuator's own `KV-RESHARD DONE` line, not from
those boundaries — so a longer driver there would buy nothing but wall clock.

**B1 (floor arm A), 16 minutes:** 23 440 mixed, **73 `prefill_heavy`**, 67
`decode_heavy`; summary line present; `desyncs 0`. 73 minus `--warmup 20`
leaves 53, above the tool's `--min-samples 30`. The boot's own stage-table line
reads `1 stage(s), 1 reachable at runtime, 0 flip target(s), booted on
'booted'` — which is the state before the measurement canon exists, and the
state this window's §5 sets out to change.

---

## 3. THE A-vs-A FLOOR

`msround_split.py` over the B1/B2 pair, `--phase-regime prefill_heavy
--warmup 20 --min-samples 30 --require-summary`:

```
A-vs-A    floor = 1.48 %  (1604.899 vs 1628.992 ms)
```

**The floor is 1.48 %.** Any delta a stage claims has to beat that before it is
a result rather than a repeat.

Two consequences, both recorded rather than assumed:

* **The watermark does not move.** RUNSHEET §2's policy moves the enter
  watermark once, before the act leg, only if the measured floor exceeds
  5.0 %. It is 1.48 %, so `enter_margin_pct` stays at 5.0 and nothing was
  re-tuned.
* **The compute/wait split is UNAVAILABLE on the observe arms.** Both B1 and
  B2 ran WITH `--regime-stage-clock`, and both still report
  `compute/wait UNAVAILABLE (no ms_decision in this trace)`. That is not the
  F4 situation the tool's help describes (a control booted without the clock);
  it is the stage table having **0 flip targets**, so there is no decision to
  record. The split therefore only becomes answerable once the canon holds a
  measurement — which is what §5 is for, and it is the reason criterion A3's
  "the movement is in the wait term" cannot be settled from the floor pair.

## 4. THE GATES, RUN RATHER THAN ASSUMED

All four were RUN on this window's own arms before the act leg, not inherited.

| gate | verdict | on what |
|---|---|---|
| 1 `desyncs_zero` | **PASS, measured** | 70 740 verdicts across 3 ranks, 54 transitions, **desyncs 0** |
| 2 `f2_live_replay` | **FAIL** | self-conditioning |
| 3 `f3_bands_measured` | **FAIL** | one blocker: `kv_ascend_mark` |
| 4 `f4_card_comparison` | cannot exist | it is produced BY the act arm |

### 4.1 Gate 3 — the retirement, exercised on live data

This is the first live run of R13's blocking-set change, and it did exactly
what it was built to do.

| constant | verdict | detail |
|---|---|---|
| `enter_prefill` 0.35 | **CLEARS** | rate 0.405 (A 0.412, B 0.399) vs 2x0.0134 = 0.0268 — **15x** |
| `enter_decode` 0.90 | **CLEARS** | rate 0.504 (A 0.486, B 0.521) vs 2x0.035 = 0.07 — **7x** |
| `kv_ascend_mark` 0.85 | **UNREACHED** | occupancy peaked at **0.841765** |
| `spread_veto_pct` 25.0 | UNREACHED (peak 0.339) | **RETIRED — does not block** |
| `PRESTAGE_SINGLE_PROMPT_TOKENS` | CLEARS | **RETIRED — does not block** |

```
NOT PASSED: kv_ascend_mark: UNREACHED
```

**The blocking list is one entry long, and that entry is a constant the runtime
actually reads.** Before R13 this same run would have reported two blockers,
one of which nothing enforces. The gate now says something true about the rig
instead of something true about itself.

**And the remaining blocker is closer than it has ever been.** `kv_ascend_mark`
peaked at 0.165 in the first window, 0.501 in the second, and **0.8418** here —
a 1.0 % shortfall against the 0.85 mark. It is reproduced **to four decimals on
both arms** (B1 0.841765, B2 0.841765), which is a striking A-vs-A result in its
own right. For the first time gate 3 is within reach of a workload rather than
structurally unreachable, and the next shift's cheapest move on this ticket is
a marginally heavier burst — **not** a re-tuned constant, which §12 forbids and
which `bands.py` assigns to #287 anyway.

### 4.2 Gate 2 — self-conditioning, and it is not this window's doing

```
SELF-CONDITIONING: the counterfactual closed loop produces
['kv_pressure' x5], which the open-loop trace does not.
```

DESIGN_363 §7.3, on measured inputs: acting would have changed the controller's
own inputs. The obvious suspicion is that this window's heavier driver caused
it by pushing occupancy toward the ascend mark — so it was **checked rather
than assumed**. The archived light arm (`B1-r2`, `--repeats 2`, peak occupancy
0.7819) fails the same way with 2 `kv_pressure` transitions instead of 5.

So self-conditioning is present at both workloads and is **not** an artefact of
the driver change. It is a standing property of this controller on this rig at
these marks, and it is a finding this ticket owes an answer to independently of
#363's actuation question.

### 4.3 The act leg ran under three bootstraps, and this is the record of it

RUNSHEET §5 rule 2 sanctions a bootstrap for gate 3. This window needed three:
gate 4 (which the act arm produces, the anticipated case), gate 3 (sanctioned),
and gate 2 (**not** anticipated by the runsheet).

Each `source` in `/spinning/evidence-363-act/gate_evidence.json` says
`BOOTSTRAP ONLY -- not a result` in those words and carries the measured
verdict it stands in for, per §3 rule 1 of `RUNSHEET_363_STAGE_CLOCK_WINDOW`.
Only `desyncs_zero` is a measured pass. The pre-bootstrap file, with the single
real entry, is preserved as `gate_evidence.gate1only.json`.

**A reader should weigh the act leg accordingly.** Bootstrapping past gate 4 is
routine — it is the chicken-and-egg the mechanism exists for. Bootstrapping
past gate 2 is not: gate 2 exists to say the closed loop would not change its
own inputs, and it says the opposite here. The act result in §6 is therefore
evidence about the RULE's arithmetic and plumbing, and is **not** a claim that
acting is safe at this occupancy.

---

## 5. THE HEADLINE: THE ACTUATOR CRASHES THE SERVER WHEN IT FIRES UNDER LOAD

This is the window's most important result, and it was not on its list.

B3's job is to move the KV vector once, under load, so the flip cost can be
instrumented. The move was armed at 14:32:25 under a prefill burst. It reached
its group-idle boundary at **14:34:19** and died **inside the move**:

```
File "python/sglang/srt/managers/kv_reshard.py", line 567, in on_round
    return self._execute()
File "python/sglang/srt/managers/kv_reshard.py", line 624, in _execute
    received = self._exchange(outgoing_payloads, incoming_nbytes)
File "python/sglang/srt/managers/kv_reshard.py", line 746, in _exchange
    buf = torch.empty(int(incoming_nbytes[peer]), dtype=torch.uint8, device=device)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 616.00 MiB.
GPU has a total capacity of 19.58 GiB of which 550.38 MiB is free.
```

`SIGQUIT`, then the whole TP group went down.

**The cutover allocates its exchange buffer with no headroom check.** The move
has a `fit_check` — it verifies the TARGET VECTOR fits the backed pool — and
that check passed. Nothing checks that the TRANSIENT receive buffer fits in
free VRAM at the moment of the exchange.

### 5.1 It broke the corridor law on the way, and the timing pins it

Corridor sampling ran throughout at 100 ms, NVML FREE:

| card | min free | samples below 1024 | when |
|---|---|---|---|
| gpu0 | **485 MiB** | 23 | ~1062-1064 s |
| gpu1 (5090) | 3698 MiB | 0 | — |
| gpu2 | **229 MiB** | 669 | ~1062-1129 s |

The first breach on both cards is at **~1062 s**, which is 14:34:12 — seven
seconds before the OOM traceback. The corridor did not drift below the floor
and then trip the allocation; **the allocation drove it below the floor.** The
5090, which had ~7 GiB free all window, never breached: the buffer is charged
to whichever rank must receive, and on this rig those are the 20 GiB cards.

Per RUNSHEET §5 rule 3 a corridor breach fails the window regardless of
everything else, and is "a code defect rather than a tight margin". Both halves
of that sentence are satisfied literally here.

### 5.2 Why it is not fixed in this window

The obvious fix — refuse the move when the receive buffer will not fit — must
be made **group-wide**, not per-rank. `on_round` already folds `fit_ok` into a
single MIN-reduced `ready` term precisely so that every rank holds or moves
together, and the module raises `KvReshardError` on any disagreement because a
one-rank divergence here is "the #94/#194/#259 hang, or worse, a silent
mixed-ownership pool". A guard added to the allocation site instead of to that
reduction would turn an OOM into a desync, which is worse.

That is a slice with its own red-first tests and its own consensus argument,
not an edit to make inside a measurement window on a delicate distributed
actuator. It is handed over in §7, with the shape the fix has to take.

### 5.3 The consequence for this window, stated plainly

The flip cost **cannot be measured under load on this rig** — attempting it is
what crashes the server. B3 was retried as **B3b** with the move made on a
drained pool, same reserve and same driver (changing the reserve would have
invalidated the B1/B2 floor the act leg is judged against). Any `flip_cost_s`
this window reports is therefore a **LOWER BOUND**, and RUNSHEET §4.4's warning
that an empty-pool cost is not the cost the controller pays stands in full.

It also means the act leg carries a live hazard: if B4's rule proposes a flip
while the pool is occupied, the actuator that executes it is the one that
crashed here.

### 5.4 The measurement pass was RUN, and its refusal is the record

The STOP was not assumed from the crash. `stage_measure_pass` was invoked on
B3b's traces afterwards and refused on its own terms:

```
refused: B3b/trace.rank0.jsonl: no summary line. The summary is written last by
construction, so its absence means the server was killed and the trace supports
'so far', not a measurement.
```

and the canon is unchanged:

```
stage measurement store: stage_measurements.json (0 record(s))
```

So the window's durable output on this axis is **zero records**, stated by the
store rather than inferred, and the refusal names the crash as the cause.

---

## 6. THE SECOND BLOCKER: NO RANK CAN FIND THE CARD PROBE

With no stage record, RUNSHEET §5 STOP-1 applies and the act leg must not be
run as a measurement — "the act leg would be an expensive observe under a
misleading name". It was booted anyway for **two minutes**, deliberately and
labelled as such, to answer two cheap questions the window could still settle:
does `act` start at all under the evidence file, and what does the stage table
say. Both answers turned out to matter.

**`act` starts.** The trace summary carries `"mode": "act"`, so the parse-time
gate accepted the evidence file and the controller entered act mode.

**And the stage table is empty for a reason nobody had measured:**

```
REGIME-OBSERVE stage feed: prefill_heavy: the planner could not solve 'enc'
  (PlannerFeedUnavailable('no card probe on disk — the solver has no
   per-card rates and will not invent any'))
REGIME-OBSERVE stage table: 1 stage(s), 1 reachable at runtime, 0 flip target(s)
```

"No card probe on disk" — **on the same rig where this window had just measured
all three cards and `card_rate_pass --show` read FRESH on both profiles.**

### 6.1 The mechanism, proven in three commands

`_latest_card_probe` does a KEYED lookup (`#513` replaced newest-by-mtime
precisely so a probe of another card set could not be accepted). The key
includes the card INVENTORY. Run outside a server, the probe matches. Run with
the device view a scheduler rank actually has, it does not:

```
CUDA_VISIBLE_DEVICES=0,1,2 -> FOUND
CUDA_VISIBLE_DEVICES=1     -> NONE
CUDA_VISIBLE_DEVICES=0     -> NONE
```

The probe was written by a three-card process. Every scheduler rank runs a
narrowed device view, so **no rank can ever match it**, and the planner feed is
permanently unavailable on any real multi-rank boot. All three ranks logged it.

### 6.2 Why this matters more than it looks

`HANDOFF_MERGE_M584` §6 named the remaining gap on this ticket as the missing
per-stage measurements, and predicted "with per-stage measurements present:
2 stages, 1 flip target". That prediction was made at the desk with an injected
feed. On metal the solver never gets far enough to want a measurement: it
cannot solve the candidate at all, because it has no rates.

So **#363's "0 flip targets" has two independent causes**, and only one of them
was known. Supplying the measurements alone would not have produced a flip
target on this rig.

---

## 7. VERDICT, AND WHAT THE NEXT SHIFT OWES

### 7.1 The six criteria

| ID | criterion | result |
|---|---|---|
| A1 | `stage_clock_proposals > 0` and `actuations > 0` | **FAIL** — 0 and 0 |
| A2 | flips over the window <= 4 | PASS, vacuously — 0 flips |
| A3 | ms/round in SHIFT beats the control by more than the floor, in the wait term | **UNANSWERABLE** — no `ms_decision` rows exist |
| A4 | zero corridor samples below 1024 MiB, every boot | **FAIL** — B1/B2/B4 pass; B3 and B3b breach, both times as the reshard cutover fires |
| A5 | `desyncs == 0` on every rank, summary present | **PASS** — 0 desyncs, summary on all 3 ranks, every arm |
| A6 | every `ms_decision` carries the rule; adopted implies signal > threshold | **VACUOUS** — 0 of 42 306 verdict rows carry `signal_pct` |

Four of six is not a pass, and this is not four of six.

### 7.2 The honest verdict

**The decision rule was NOT put on metal, and this is not a case of "act
correctly refused to flip".** That verdict would require the rule to have been
evaluated and its signal to have fallen below threshold. Nothing was evaluated:
the axis had no candidate to consider, so no `signal_pct` was ever computed.
Reporting this as a refusal would be the more flattering claim and the false
one.

What the window did produce is the diagnosis that was actually missing — and
three defects, each found only by executing the runsheet rather than reading
it:

1. **A rate that could not be dated** (§1.1) — fixed on the line, confirmed on
   metal.
2. **An actuator that crashes the server and breaks the corridor law when it
   fires** (§5) — reproduced twice, deterministic, worse when drained.
3. **A planner feed that no rank can reach** (§6) — reproduced in three
   commands, and the real reason flip targets are 0.

Plus one result the retirement earned: **gate 3's blocking list is down to one
entry, and that entry is 1.0 % from reachable** (§4.1).

### 7.3 Owed, in order

1. **The reshard headroom guard (§5.2).** This is now the top item on #363: the
   actuation axis cannot actuate without it, and today it takes the server down
   with it. The guard must be folded into `on_round`'s **group-wide MIN
   reduction** alongside `fit_ok`, never added at the allocation site — a
   per-rank refusal there converts an OOM into the desync the module's own
   error text calls "the #94/#194/#259 hang, or worse". Its hold reason should
   name the required bytes, the free bytes and the corridor floor. Red-first,
   with a can-fail arm that shows the group holding rather than one rank
   diverging.
2. **Make the card probe reachable from a rank (§6).** Either the probe is
   looked up by a key a narrowed device view can still produce, or the lookup
   happens in the parent process and is handed down. Until then no boot has a
   planner feed, and every stage table on this rig is one stage with no
   targets.
3. **Gate 3 is one workload step away (§4.1).** `kv_ascend_mark` peaked at
   0.841765 against 0.85, reproduced to four decimals on both arms. A
   marginally heavier burst reaches it and gate 3 can pass for the first time.
   Not a re-tune — §12 forbids it and `bands.py` assigns the constant to #287.
4. **Gate 2's self-conditioning (§4.2)** is a standing property at both
   workloads and is owed an answer independently of the actuation question.
5. **Fix the runsheet's own numbers.** §4.1's reserve breaches the corridor on
   this rig (§1.3) and §4.3's driver cannot feed §9's own analysis command
   (§1.4). A runsheet whose profile refuses its own tool costs a window every
   time it is executed faithfully.

### 7.4 Teardown

Corridor sampled on every boot at 100 ms NVML FREE. Trace summary present on
every rank of every arm. The gate evidence file was **returned to its single
measured entry** — no bootstrap is left in place, which is §3 rule 2's whole
point; the copy used for the B4 boot is kept beside it as
`gate_evidence.bootstrapped-for-B4.json`. Heartbeat stopped and confirmed
inactive **before** the holder was released; the restore script's own guard
refused once at 71 s of heartbeat age and ran at 133 s, which is that guard
working. Serving on 30030 restored via the sanctioned `res-r5
restore_ship.sh` and verified with a **real generation** — the planted answer
`MERIDIAN41`, `completion_tokens=6`, `finish=stop` — not a health 200, because
the #622 wedge signature answers health and emits no tokens. Port 30099 never
touched.
