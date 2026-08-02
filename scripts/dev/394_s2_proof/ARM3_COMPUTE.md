# Arm 3 — link-proportional expert COMPUTE placement (#394 slice 3 / task #439)

The two arms this directory shipped with move BYTE ownership. Arm 3 moves the
assignment: which rank executes which expert. It is the arm ANALYSE_393 §7.3
called Path A′, and it is the first arm in this window whose predicted per-rank
H2D delta is NOT null.

    ARM=equal          baseline, pre-#394 plan field for field
    ARM=proportional   slice 2: cold BYTES follow the links, compute does not
    ARM=compute        slice 3: cold COMPUTE follows the links   <-- this file
    ARM=compute-cal    slice 3 with the traffic coefficients measured on arm 1

## The one flag

    --rank-moe-ratio link

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
per-rank cold-traffic coefficient, 1.0 unless a prior boot calibrated it.

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
The numbers for the plan the recipe actually resolves are in "Confirmation
window" below. Read the resolved plan off the boot log, as this file has always
said, and use the section that matches it.

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
resolves — see "Confirmation window". It falls short of BENCH_394's 1.54x "ideal
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

**Report the band, not one end of it.** On this base plan the honest statement
is 1.36x uncalibrated / 1.58x calibrated on the transfer term, against
BENCH_394's 1.54x ideal-placement reference; on the plan the recipe resolves it
is 1.39x / 1.45x. Either way, running both sub-arms is what turns a band into a
measurement; running only `compute-cal` would report a number whose calibration
came from the arm it is being compared against.

## Confirmation window (the next boot window, spec)

One window, three boots, in this order. The three defects the 2026-08-02
battery found are fixed at the desk and are unvalidated on hardware until this
window runs: **BOOT-PENDING**.

Inputs, all read off that battery's own record
(`/spinning/gpu-battery-results/2026-08-02_439_arm3/RESULTS.md`) rather than
assumed:

| input | value | where it came from |
|---|---|---|
| base plan | `30407,19080,19080` | resolved `--rank-tp-ratio` in `boot_equal.log` |
| base expert counts | 114 / 71 / 71 of 256 | `partition_units(256, base)` |
| resident fraction | `0.485,0.42,0.42` | the launch flag, unchanged across arms |
| link weights | 14.42 / 6.45 / 13.41 GB/s | measured card probe, `provenance=measured` |
| measured H2D | 888.6 / 557.7 / 598.2 GiB | equal arm's #390 dump |
| baseline transfer | 66.2 / **92.8** / 47.9 s | H2D / link; tp1 (x4) is the clock |
| traffic coefficients | `1.0561,0.9379,1.0060` | `cold_traffic_coefficients_from_measurement` on the four rows above |
| same-boot A-vs-A floor | CV 1.19 %, spread 2.35 % | equal arm, 3 x 450-token generations |

Predictions for THIS base plan (all four inputs above fed through the module's
own functions; the group H2D total is held at the measured 2044.5 GiB, since
the same tokens reach the same experts and only ownership moves):

| arm | vector | expert counts | predicted H2D (GiB) | transfer (s) | clock |
|---|---|---|---|---|---|
| `equal` | 30407,19080,19080 | 114 / 71 / 71 | 888.6 / 557.7 / 598.2 | 66.2 / 92.8 / 47.9 | 92.8 |
| `compute` | 160,79,119 | 114 / 57 / 85 | 895.5 / 355.7 / 793.3 | 66.7 / 59.2 / 63.5 | **66.7 = 1.392x** |
| `compute-cal` | 258,135,197 | 112 / 59 / 85 | 860.0 / 384.7 / 799.8 | 64.0 / 64.0 / 64.0 | **64.0 = 1.450x** |

**The band to confirm is 1.39x / 1.45x on the transfer term, not 1.358x /
1.584x.** The latter belongs to the `400,256,344` worked example above. The
window's job is to measure this band, not to reproduce a number from a
different plan.

Discipline for the window:

* **One boot per arm.** Three boots: `equal`, `compute`, `compute-cal`. Each
  ~7 minutes of load; nothing here is worth a re-boot to re-read.
* **A-vs-A floor first, in the `equal` boot**, before any delta is quoted. The
  2026-08-02 value (CV 1.19 %) is the expectation, not a substitute — a floor
  measured in another window does not cover this one.
* **`--rank-auto-reserve-mib auto`** (`RESERVE_MIB=auto bash run_arm.sh ...`).
  The pinned `2200,1400,1400` left both 3080s at ~515 MiB free — above the
  400 MiB corridor floor, but by 115 MiB, and that is the exact margin the
  residency defect overran. `auto` derives the reserve per card instead of
  carrying one window's tuning into the next. Same value on every arm of the
  window; a reserve that differs between arms is a second treatment.
* **Same recipe, same fraction, same reserve on all three arms.** The solve
  holds the resident mass fixed against the base plan, so changing the fraction
  between arms changes the treatment rather than the measurement.
* **Check the arm identified itself** before reading anything else:
  `facts_<arm>.txt` must show `moe_compute_policy=link-proportional` (or
  `-calibrated`), the solved vector, and — on the two compute arms — the
  per-rank `resident fraction held at the base plan` lines. A compute arm
  without those lines is not VRAM-neutral and its corridor is not arm 1's.
* **Corridor** per card >= 400 MiB free, sampled at 5 s into `corridor.csv`.

```
export RUN=/spinning/gpu-battery-results/$(date +%F)_439_confirm
export REPO_ROOT=/spinning/htsglang VENV=/spinning/htsglang-gpu/.venv
export WT=<the worktree> PORT=30439 RANK_GPU_ID=<from preflight.sh> RESERVE_MIB=auto
bash "$WT/scripts/dev/394_s2_proof/preflight.sh"          # must print PREFLIGHT OK
bash "$WT/scripts/dev/394_s2_proof/run_arm.sh" equal
bash "$WT/scripts/dev/394_s2_proof/run_arm.sh" compute
SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS=1.0561,0.9379,1.0060 \
  bash "$WT/scripts/dev/394_s2_proof/run_arm.sh" compute-cal
```

Re-derive the coefficients from THIS window's own `equal` arm if its H2D
differs from the row above by more than the A-vs-A floor; a coefficient
measured on one recipe is not a property of the rig.

## What must be read out, per rank

The slowest-rank rule governs every line: a group mean hides exactly the effect
this arm exists to move.

1. **Per-rank H2D** (`totals.h2d_bytes` in the #390 dump). The primary readout.
   Unlike arms 1 and 2, a NULL delta here falsifies the arm.
2. **Per-rank ms/round.** Transfer is part of a ~135 ms/token decode on this
   recipe, so the end-to-end effect is smaller than the transfer-term ratio.
   Report both, and report the per-rank split (compute vs wait) from
   CollectiveClock — a rank that stops being the clock should show its wait time
   move, not vanish.
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
* Arms in one window, same recipe, same reserve (`2200,1400,1400`), same
  `--rank-moe-resident-fraction 0.485,0.42,0.42`. The solve holds the resident
  mass fixed, so changing the fraction between arms changes the treatment.
* `--disable-cuda-graph`, as the published baseline for this configuration does.

## Corridor and preflight

VRAM-neutral by construction, so the corridor is arm 1's: per-card free
>= 400 MiB, no registered posting wasting > 1.5 GiB net. Host DRAM is where the
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
bash scripts/dev/394_s2_proof/preflight.sh          # must print PREFLIGHT OK
ARM=equal   bash scripts/dev/394_s2_proof/boot_ab.sh
# bounded curl -m readiness loop, then the bench-length generations
python3 scripts/dev/394_s2_proof/read_arm.py <run> equal
ARM=compute bash scripts/dev/394_s2_proof/boot_ab.sh
python3 scripts/dev/394_s2_proof/read_arm.py <run> compute

# calibrated sub-arm: coefficients from the EQUAL arm's own dump
python3 -c "
from sglang.srt.layers.moe.expert_compute_placement import (
    cold_traffic_coefficients_from_measurement as c)
print(','.join(f'{x:.4f}' for x in c([400,256,344],[0.485,0.42,0.42],[H0,H1,H2])))"
ARM=compute-cal SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS=<that> \
  bash scripts/dev/394_s2_proof/boot_ab.sh
python3 scripts/dev/394_s2_proof/read_arm.py <run> compute-cal
```

`H0,H1,H2` are the equal arm's per-rank `h2d_bytes`, and `[400,256,344]` is the
base plan THAT arm ran (read it off the boot log's resolved `--rank-tp-ratio`,
do not assume it).

## Status

**The band is UNMEASURED. The arm has never served a token.**

The 2026-08-02 battery ran the `equal` baseline in full (that half is real, and
is where every measured input above comes from) and could not boot `compute` at
all. Three defects, all found on hardware, all fixed at the desk since:

| # | defect | fix | falsifier |
|---|---|---|---|
| 1 | the resolver read the resident fraction as `1.0` — it runs in the launcher, before the ServerArgs reach the runtime context, and the flag source swallowed the resulting error | the resolver hands its own `server_args` down to `resident_fraction_vector`; the env/flag cross-check is unchanged | `TestTheResolverReadsTheLaunchFLAG`, can-fail = restore the context-only route (5 tests fail) |
| 2 | the arm was not VRAM-neutral: residency was sized off the SOLVED expert count, +19.5 % on tp2 → OOM during staging. Correcting the fraction feeds back into the solve (a fixed point the spec did not address) | residency held at the pre-link base plan through a DERIVED sizing fraction on a channel the solve never reads — see the NOTE above | `TestResidencyIsHeldAtTheBasePlan`, can-fail = drop the correction (tp1 25≠31, tp2 37≠31) |
| 3 | the battery's driver set `ARM` without exporting it, so every arm booted the baseline; `boot_ab.sh` defaulted to `equal` rather than refusing | `run_arm.sh` is in the repo and exports; `boot_ab.sh` refuses an unset arm and has a `DRY_RUN=1` mode | `TestTheArmHarnessCarriesTheArm`, can-fail = restore the `equal` default (the unexported-arm case fails) |

Hermetically tested: `test/registered/unit/layers/moe/test_expert_compute_placement_439.py`,
76 tests + 141 subtests, including an execution smoke of the full resolver path
with the hardware facts injected through `SGLANG_RANK_CARD_UUIDS` +
`SGLANG_MOE_HOST_SHARD_RATIO`, and seven proven can-fail arms (solve ignores the
links; resident mass allowed to float; launcher call removed; worker refusal
removed; plus the three above). None of it has been observed on hardware.
The standing DESK-WRITTEN risk label applies until the confirmation window
above runs. The catalog-first analysis behind the design is
`docs/dev/ANALYSE_439_expert_compute_placement.md`; the battery record is
`/spinning/gpu-battery-results/2026-08-02_439_arm3/RESULTS.md`.
