# Measurement ticket: #363 intra-phase stage actuator

STATUS: WRITTEN, NOT RUN. Everything below is desk work validated by hermetic
unit tests only. No line of the intra-phase axis has executed on a GPU. This
ticket is the window that would change that; it is written so the next GPU
window can execute it without re-deriving anything.

Branch: `feat/regime-stage-actuator-363`. Worktree `/spinning/wt-363-stages`,
`PYTHONPATH=/spinning/wt-363-stages/python`.

---

## 1. What is being measured, in one sentence

That an ms/round-driven stage flip, admitted through the corridor guard,
improves the round time when the load changes regime mid-window, without
breaching the corridor and without thrashing.

## 2. Why a window is needed at all

Three of the four claims the axis rests on are UNMEASURED on this rig:

| Claim | State | Where it would be settled |
|---|---|---|
| The wait term is what a stage flip moves | ASSUMED (mechanism argument) | acceptance A3 below |
| The enter watermark (5 %) is above this rig's A-vs-A band | ASSUMED (policy number) | pre-step P2 |
| A stage flip's transient is per-load-state | MEASURED elsewhere (`pp_cut` records 956 vs 1989-3148 MiB on this rig) | reused, not re-measured |
| The gate refuses an unfunded move | UNIT-TESTED against a stub guard | acceptance A4 |

The second row is the one that most plausibly comes back wrong. If this rig's
A-vs-A band on the stage pair exceeds 5 %, the shipped enter watermark is
inside the noise and must move; the code already refuses a flip whose signal
does not clear the measured band, so the failure mode is "never flips", not
"flips on noise".

## 3. Pre-steps (cheap, do first, no load)

**P1 — a stage table with MEASUREMENTS.** The axis refuses to propose an
UNMEASURED stage (`#578` placeholder zeros are not a measured gain of zero),
so a planner-only table produces a clean run in which nothing ever happens
and the window proves nothing. Confirm before booting:

```
grep -c '"unmeasured": true' <the stage table dump in the boot log>
```

If every non-reference stage is unmeasured, STOP: the window cannot pass and
the missing work is #584's measurement pass, not this ticket.

**P2 — the A-vs-A band for the stage pair.** Back-to-back repeats of the SAME
stage, warmup discarded, >= 10 s per arm, clock/power fixed. This is the
number the enter watermark has to sit above. Record it; if it exceeds 5 %,
raise `MsStageDecider(enter_margin_pct=...)` above it rather than lowering
the band.

**P3 — the transient census for every stage in the acting set.** Without it
every flip is REFUSED by name (this is deliberate: an unpriced term reads as
free memory). Boot once per stage with `SGLANG_TRANSIENT_CENSUS=1` and
`SGLANG_RESIDENCY_CENSUS_DIR=<dir>`, under BOTH load states the window will
visit — deep-prefill and decode-heavy. A partial set is refused loudly by
`pp_cut_calibration`; that refusal is correct and is not a bug to work around.

## 4. The window

Boot (single node, TP as configured on the rig; the axis is scoped to pure TP
and inherits every #656 restriction):

```
--regime-controller act \
--regime-gate-evidence <the existing evidence file> \
--regime-stage-clock \
--regime-trace /spinning/traces/363-stageclock-rank{}.jsonl
```

Load profile — ONE window, regime shifts in the middle, because the whole
claim is about a shift:

| Phase | Duration | Shape |
|---|---|---|
| Settle | 120 s | decode-heavy, bs 4, short prompts. Let the clock's window fill and the incumbent hold. |
| SHIFT | 300 s | deep-prefill burst: long prompts, few outputs. This is the regime change. |
| Return | 300 s | decode-heavy again, same as settle. |

Do not restart between phases. The point is the TRANSITION, and a restart
replaces the thing being measured with two steady states.

## 5. Acceptance

All five, and each one names where its number comes from. A window that
passes four of five has not passed.

**A1 — IT MOVED.** `stage_clock_proposals > 0` and `actuations > 0` in the
observer summary, with at least one actuation inside the SHIFT phase. Read
off the trace's `ms_decision` and `actuated` fields, not off a log line.

**A2 — IT MOVED BACK, AND NOT MORE.** Total flips over the whole window
`<= 4`. Two are structural (into the prefill stage, back out). More than four
in a 12-minute window with two genuine regime changes is thrash, and the
`flip_cost_s` of the pair is what makes it expensive.

**A3 — ms/ROUND IMPROVED.** Compare the mean ms/round over the last 120 s of
the SHIFT phase against the same measure from a control window run WITHOUT
`--regime-stage-clock`, everything else identical, same boot. The controlled
comparison is required: absolute ms/round across the shift is dominated by
the load change, so an uncontrolled improvement number measures the workload,
not the controller. Report compute and wait separately — the mechanism claim
is that the WAIT term is what moved, and if the win is entirely in compute
the arithmetic in `regime_ms_clock` is crediting the wrong term.

**A4 — ZERO CORRIDOR BREACHES.** NVML free per card, sampled at 100 ms,
minimum over the window `>= 1024 MiB` on every card. A time-series minimum,
not a boot snapshot, and the NVML FREE column, never `total - used`. One
breach fails the window regardless of A1-A3: the admission gate exists
precisely to make this impossible, so a breach means the gate was bypassed
rather than that the margin was tight.

**A5 — NO DESYNC.** `desyncs == 0` in every rank's trace summary, and the
summary line present (its absence means the run did not end cleanly, so
"zero so far" is all the trace supports).

## 6. What each failure would mean

| Symptom | Most likely cause | Next move |
|---|---|---|
| Nothing ever proposes | stage table unmeasured, or signal inside the band | check P1, then P2; the refusal reason in the trace names which |
| Proposes, never actuates | admission refusing | read `admission.last.reason` — it names the load state, both price terms, and whether the refusal was local or group |
| Actuates, ms/round worse | the wait term was not what moved | A3's split answers it directly; the conservative arithmetic should under-promise, so a REGRESSION points at the flip cost, not the prediction |
| Flips more than 4 times | hysteresis windows too short for this rig's boundary cadence | raise `exit_window` before touching the watermarks; dwell (`DwellGate`) is the other lever and is deliberately separate |
| Corridor breach | the gate was bypassed | this is a code defect, not a tuning problem — the flip path must not reach an actuator except through interlock 5 |

## 7. Explicitly out of scope

- Combination with PP/DP/EP. The axis inherits #656's scope.
- Tuning the watermarks to make the window pass. If P2 says the band is wider
  than the mark, the mark moves ONCE, before the window, and is recorded as a
  policy number with its measurement — not adjusted afterwards until the
  result is agreeable.
- The weight-cut axis. It has no runtime actuator; a stage differing in it is
  reported and never selected, unchanged by this work.
