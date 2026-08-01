# #375 probe runsheet — prefill-heavy, higher batch

**~15 minutes, one boot pair.** Turnkey; the scripts have been smoke-run
(§5) so the window is spent measuring, not debugging.

## Why this operating point

The #350 window measured the decode point at bs=1 and found **no divergence**:
both objectives installed the identical vector `16,1,1`, so the A/B was
structurally an A-vs-A. Two reasons, both of which this probe changes:

1. **Decode at bs=1 is memory-bound on KV streaming**, so there is little
   compute-power asymmetry for an energy objective to exploit. The 5090 idles
   at 45.9 W against 3080s at 57.2 and 105.4 W — that asymmetry has room only
   when the cards are actually working.
2. **The admissible set had exactly one member.** Under the default perf-tune
   every concentration candidate was rejected — first unfundable, then by the
   decode-knee guard — so the objective had nothing to choose between. Correct
   behaviour (guards bind before the objective), but it makes the A/B empty.

So: **prefill axis, bucket 8, prose workload, `--rank-perf-tune phase-prefill`**
(knee advisory, which in the #350 window opened the admissible set to 7 priced
candidates). If the objectives still pick the same vector here, that is a real
negative result about this rig rather than an artefact of a saturated guard.

## Run

Claim the cards first (`/spinning/gpu-arb/`: holder + 60 s heartbeat + FREI +
0 MiB). The script does not arbitrate.

```bash
cd /spinning/wt-375
PROBE=1 scripts/energy350/tokj_validation.sh
```

That is the whole probe. It boots two arms differing **only** in
`--objective`, measures each with the #146 harness, and prints the verdict.
Artifacts land in `/spinning/gpu-battery-results/<date>_375_tokj/`
(`result.json`, `VERDICT.txt`).

Optional: `BUCKET=16` for a second point if the window has slack. Do not add
axes — one variable at a time is what makes the delta attributable.

## Minimal points

| | |
| --- | --- |
| arms | 2 (throughput, energy) — one boot each |
| bucket | 8 (16 only if time remains) |
| workload | prose, 1 |
| axis read | **prefill** |
| context | 8192 |

Expected wall time: two boots at roughly a minute each plus measurement —
the #350 window's arms took ~43 s apiece end to end.

## Reading the verdict

`run_ab.py` prints one of four outcomes and exits 0/1/2:

* **GREEN** (exit 0) — energy arm has fewer J/token *and* fewer tok/s. The
  predicted trade reproduces.
* **AMBER** (exit 1) — energy arm won *both* axes. **Not a pass.** Then the
  throughput arm was not the throughput optimum; investigate that before
  quoting anything.
* **RED** (exit 1) — energy arm did not reduce J/token here. Do not report the
  objective as validated at this point.
* **INCONCLUSIVE** (exit 2) — an arm failed. Nothing is inferred from a number
  that does not exist; re-run that arm.

**A fifth outcome the exit code cannot express, and the one #350 actually
produced: both arms install the same vector.** Check it before reading the
deltas —

```bash
grep -oE "rank-mlp-ratio [0-9,]+" /tmp/energy_boot_31350.log | tail -2
```

Identical vectors mean the A/B was an A-vs-A and the deltas are the noise
floor, not a result. Report it as *"no divergence at this point"*, not as
GREEN. The #350 floor for reference: **0.638 tok/s and 0.240 J/tok**.

## Provenance duty

Confirm the boot log line `objective=energy: planning for J/token, measured
power anchors` before trusting the energy arm. If any card resolves to
`estimate`, the run is estimate-grade and the report must say so — that is a
finding, not something to work around by dropping the flag.

## What this does not cover

One operating point on one rig. Not the batch-size curve, not other models,
not other context lengths. A GREEN here means the objective discriminates
*somewhere*, which is the specific thing #305 cut 4 is gated on — it does not
mean the objective is generally useful.

## Notes for later

`compare_tokj.py` (the text-report parser from the phase-2 turnkey) is
superseded for this flow: `run_ab.py` verdicts directly from the structured
result and cannot mis-parse a reworded harness line. It is kept for reading
older text reports.
