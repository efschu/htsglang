# #328 — r8-E re-measure with the corrected window, plus the content gate

Turnkey. Prepared on the desk; **not run** here (the cards belong to the #370
agent). A later window executes it in minutes.

## Read this first: what is already done

**Posten 1 of #328 has already been measured.** The corrected-window r8-E
values are in `docs/dev/INTEGRATION_R3_VALIDATION.md`, section *"#328 Posten 1:
r8-E-Werte mit korrigiertem Fenster — und ein Befund"*:

| quantity | r8 (4403a98312) | corrected (cc522801e2) |
| --- | --- | --- |
| lane shared, no chain | 11.000 (tail-inflated) | **28.04** tok/s |
| lane shared, with chain | 15.733 (tail-inflated) | **31.33** tok/s |
| share_lane (decode def.) | 0.193 / 0.298 | **0.499 / 0.593** |
| E (r8 definition) | 1.035 / 1.140 | **1.441 / 1.534** |

So this runsheet is **not** "re-measure E from scratch". What that section
left open, and what a window should actually buy, is:

1. **The E effect of the ENGAGED policy** — n=1 is not enough; it needs
   several interleaved windows per arm in one boot.
2. **Which merge closed the submission gap** (occ_r 0.39 -> 1.00 between r8
   and that HEAD) — bisect-shaped, its own ticket.
3. **Threshold calibration** from the window labels rather than the roofline
   defaults.

Item 1 is what the script below drives, with the content gate attached.

## What changed in the method (why the old numbers were withdrawn)

Round 8 read the lane's wall-clock counters **after** stopping the serving
load, so the stop's worker join landed inside the shared window. #284 caught
it as an impossibility — duty 1.208-1.377, more occupied wall time than the
window had — and corrected the reader to count **before** stopping, naming a
duty > 1 instead of clamping it. Only SHARED windows were affected, i.e.
exactly the numerator of `share_lane`, so the old shared rates were
over-estimates and the reported loss was **larger** than r8 said, not smaller.
Both r8 and R4 now carry a retreat notice pointing here.

## Run

```bash
cd /spinning/wt-328
scripts/dual_group/r8redo/rerun_e.sh 30081
```

Claim the cards through `/spinning/gpu-arb/` first (ANFRAGE, 60 s heartbeat,
FREI, cards back at ~0 MiB). The script does not arbitrate.

It boots once with the r8 recipe (NEXTN, both lanes, `--dual-group-lane-spec`),
runs the corrected-window phases interleaved so each arm gets several windows
in one boot, then runs the content gate over the two control reports.

## The content gate

`scripts/dual_group/chain_quality_gate.py` decides whether turning the chain
on made the OUTPUT worse. Two rules it exists to encode:

- **Not text identity.** Greedy speculation only reproduces the greedy
  trajectory if the verify forward is bit-identical to the decode forward, and
  it is not. A near-tie flip that still emits the determined sequence is not a
  regression (#360/#365).
- **Not a pre-registered constant.** The band is measured on the same boot as
  `max(|ref_a - ref_b|, |cand_a - cand_b|)` — both arms' A-vs-A repeat, wider
  wins. A 1e-3 constant would have decided #274 by itself and decided it
  wrong.

```bash
python scripts/dual_group/chain_quality_gate.py \
  --reference "$OUT/nospec.json" --candidate "$OUT/spec.json" --json
```

Exit codes: `0` GREEN, `1` RED, `2` VOID. **VOID is not a pass** — it means an
arm did not hold still, a prompt had no scorer, or a repeat was missing, and
no constant stands in for a band that was not measured. Do not report an arm
whose gate went VOID; fix the instrument and re-run that arm.

## Accept-length discipline (#326)

Read accept length from `meta_info.spec_accept_length` (and
`spec_verify_ct`), never from the Prometheus EMA — `spec_ema_accept_len` is
not the accept length. The control reports already carry the right field.

## Harness duty 7

The report must carry the per-position accept curve and a known
content-baseline column at high K, not just an aggregate. If the window only
produces aggregates, it has not discharged duty 7 and the numbers are not
quotable as a curve.
