# #350 phase 2 — tok/J validation runsheet

Turnkey. Prepared on the desk; **not run** (the cards were held by the #366
measurement agent). The next GPU window can execute this in minutes.

## What it tests

The solver's energy objective predicts a **trade**, not a free win:

| arm | plan | prediction |
| --- | --- | --- |
| `--objective throughput` | time-optimal vector | more tok/s, more J/token |
| `--objective energy` | energy-optimal vector | fewer J/token, fewer tok/s |

On the desk fixture the two optima genuinely differ — `(10,1,1)` vs `(12,0,0)`,
i.e. **9.8 % fewer J/token at 18.2 % slower**. The GPU run asks whether that
divergence survives contact with the real rig.

## How each arm's plan is selected (updated for phase 4)

`--objective` is now honoured by the BOOT itself: `--rank-tp-ratio
auto-performance` selects the admissible MLP vector with the lowest predicted
J/token. The two arms are therefore the same boot with one flag changed --
the plan-then-boot curl the phase-3 runsheet needed is gone.

```bash
# energy arm  -- boots the energy-optimal vector directly
... --rank-tp-ratio auto-performance --objective energy
# throughput arm -- omit the flag (or pass --objective throughput)
... --rank-tp-ratio auto-performance
```

Read the chosen vector and the pricing tier off the boot log's
`auto-performance` block: it now prints `objective=energy: planning for
J/token, <measured|estimate> power anchors (...)` and a `predicted N J/token`
figure next to every candidate. Quote the tier in the verdict -- an
estimate-anchored run is estimate-grade.

If the rig cannot be priced (a card with neither an NVML calibration row nor a
TDP in the card library) the boot FAILS at parse time with a named reason. That
is the intended behaviour, not a bug in the script: run the #149 power
calibration first, or add the card's TDP. Never "fix" it by dropping the flag
and reporting the arm anyway -- that is the silent substitution the whole
design refuses.

## Before you start

1. Claim all three cards through `/spinning/gpu-arb/` (ANFRAGE, heartbeat every
   60 s, FREI at the end, cards back at ~0 MiB). This script does **not**
   arbitrate.
2. Check no other agent is mid-measurement in the arb log.
3. Budget: two cold boots plus two harness passes, ~20–25 minutes.

## Run

```bash
cd /spinning/wt-350-p2
scripts/energy350/tokj_validation.sh 31350
```

Environment overrides (all optional): `WT`, `VENV`, `MODEL_ROOT`, `MODEL`,
`OUT`. Defaults target this worktree, the shared venv, and
`Qwen3.6-27B-FP8`.

Artifacts land in `/spinning/gpu-battery-results/<date>_350_tokj/`:

- `<arm>.server.log` — boot log (the installed vector is read from here, not
  derived from the flags — the #340 trap)
- `<arm>.vector.txt` — the effective `--rank-mlp-ratio`
- `<arm>.energy.txt` — the #146 harness report (tok/s + J/token, per card)
- `<arm>.pyspy.txt` — a py-spy dump taken before each kill (standing rule)
- `VERDICT.txt` — the comparison

## Reading the verdict

`compare_tokj.py` prints one of three outcomes and exits 0/1/2:

- **GREEN** — fewer J/token *and* fewer tok/s on the energy arm. The predicted
  trade reproduces; the objective is validated at this operating point.
- **AMBER** — the energy arm won *both* axes. The energy objective is not
  wrong, but then the throughput arm was not the throughput optimum; check the
  throughput plan before quoting this result.
- **RED** — the energy arm did not reduce J/token. The solver's energy model
  does not describe this rig here; do **not** report the objective as
  validated.
- **INCONCLUSIVE (exit 2)** — a number was missing from a harness report.
  Nothing is inferred from a missing number; re-run that arm.

## What this does not cover

- Only the `dec` (decode) goal is energy-priced today; `enc` uses the same
  machinery but is unmeasured here, and the capacity goals (`maxkv`,
  `sessions`) are deliberately not energy-priceable (their per-rank terms are
  bytes, not seconds).
- One model, one context length, one operating point. The energy optimum is a
  function of the operating point; a single GREEN does not generalise across
  the batch-size curve.
- J is GPU board power via NVML — it excludes CPU and PSU losses, so it is not
  wall power (the harness says so in its own provenance line).
