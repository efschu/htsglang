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
VRAM-neutral by construction, so the reserve, the ledger and the corridor are
the ones arm 1 was validated at — and redistributes only the STREAMED remainder
in proportion to the measured link weights:

    resident_r = f_r * b_r                      # fixed
    share_r    = resident_r + (1 - sum resident) * normalise(l_r / c_r)

`l` comes from the same #394 provenance chain arm 2 uses (env > card-probe H2D
> NVML nameplate > refusal; `absent` is refused, never guessed). `c` is the
per-rank cold-traffic coefficient, 1.0 unless a prior boot calibrated it.

## Resolution point, and why it is not in the worker

The launcher resolves the symbol once, after it publishes the rank -> card
vector and before the spawn loop; the workers inherit the numbers in the pickled
`ServerArgs`. Three workers re-deriving the vector from three independent NVML
reads would put the group's expert COVERAGE on the outcome of a race — a hole or
an overlap in the ranges is a silently wrong all-reduce, not a hang. A symbolic
value that reaches a worker anyway is a hard error there
(`scheduler.uneven_family_plans`), never a fall-back to the base plan.

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

That is the 1.36x class ANALYSE_393 predicted for the short-probe mix, and it is
the number to hold arm 3 to. It falls short of BENCH_394's 1.54x "ideal
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

**Report the band, not one end of it.** The honest statement for this recipe is
1.36x uncalibrated / 1.58x calibrated on the transfer term, against BENCH_394's
1.54x ideal-placement reference. Running both sub-arms is what turns that band
into a measurement; running only `compute-cal` would report a number whose
calibration came from the arm it is being compared against.

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

DESK-WRITTEN, NEVER EXECUTED against a GPU. The solve, the launcher resolver
and the worker refusal are hermetically tested
(`test/registered/unit/layers/moe/test_expert_compute_placement_439.py`,
53 tests + 33 subtests), including an execution smoke of the full resolver path
with the hardware facts injected through `SGLANG_RANK_CARD_UUIDS` +
`SGLANG_MOE_HOST_SHARD_RATIO`, and four proven can-fail arms (solve ignores the
links; resident mass allowed to float; launcher call removed; worker refusal
removed). Every number in this file is a prediction from measured inputs, and
none of it has been observed on hardware. The standing risk label applies until
arm 3 boots. The catalog-first analysis behind the design is
`docs/dev/ANALYSE_439_expert_compute_placement.md`.
