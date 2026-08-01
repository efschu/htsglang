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

## How each arm's plan is selected (updated for phase 3)

`--objective` is honoured by the **planner**, not yet by the boot's own
`--rank-tp-ratio auto-performance` sizing. So the two arms are not "the same
boot with a different flag" — each arm's key is PLANNED first and then booted
explicitly:

```bash
# energy arm: ask the solver for the energy-optimal key (needs power anchors)
curl -s localhost:<planner-port>/api/key_solver -H 'content-type: application/json' -d '{
  "model_path": "'"$MODEL"'", "tp_size": 3, "rank_gpu_id": [0,1,2],
  "goal": "dec", "objective": "energy",
  "power_anchors": [{"idle_w": 30, "active_w": 300, "source": "measured"},
                    {"idle_w": 90, "active_w": 320, "source": "measured"},
                    {"idle_w": 90, "active_w": 320, "source": "measured"}]
}' | jq -r '.candidates[0].launch_flags | @sh'
# -> --rank-mlp-ratio a,b,c   ... boot with exactly that
```

Drop `"objective"` and `"power_anchors"` for the throughput arm. The answer
carries `mode: "energy"` and a caveat naming the J/work figure and whether the
anchors were measured or estimated; an unpriceable request comes back
`ok: false` with a named reason (never a throughput plan under an energy
label), and the script must treat that as INCONCLUSIVE, not as an arm.

Power anchors: use the NVML calibration from #149 (`power_calibration`) when
present — those are the `"measured"` tier. TDP-derived anchors from the card
library are the `"estimate-tdp"` tier and still plan, but the verdict must
then quote the run as estimate-grade.

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
