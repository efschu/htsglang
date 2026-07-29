# P2P re-probe: verdicts to re-examine after the driver update

Every placement/transport decision so far was made on a rig WITHOUT
GPUDirect P2P (chipset, all PHB, GPU0 on x4). The driver update may change
that -- in a PARTICULAR shape: the two RTX 3080 expose only the small
256-MiB BAR1 window (and how much of it is EFFECTIVELY usable is itself a
measurement), while the RTX 5090 has full-VRAM BAR. Where allreduce or
broadcast should run afterwards is OPEN; nothing below states an expected
outcome. Every row is a "check whether", answered by a measurement point
from this package.

Fill in the `after` column from `results/<date>/`; consumers of aperture
numbers use the EFFECTIVE values only, never the nominal BAR size.

## Open questions (measurement points, not predictions)

| # | Check whether... | Measurement point |
|---|---|---|
| Q1 | NCCL picks P2P at all for any pair, and over the small aperture in particular | nccl_transport_check per pair, transport_summary before/after |
| Q2 | the 256-MiB window is fully usable or effectively smaller (addressability, reservations) | capability_matrix effective_max_single_copy / effective_max_region_chunked vs nominal |
| Q3 | direct D2D beats host staging, per direction and size -- including above the window boundary | d2d_bench ladder, knee around 255/256/257 MiB |
| Q4 | both 3080 windows can be pressured simultaneously without collapsing | d2d_bench dual-window arm vs single-leg medians |
| Q5 | custom allreduce becomes constructible (can_p2p gate) and, if so, whether it is BAR-limited | capability_matrix can_access_peer + a later guarded sglang boot; #195 fix is prerequisite (see below) |
| Q6 | a broadcast topology exists that exploits the asymmetry (full-BAR 5090 vs windowed 3080s) | d2d_bench directed asymmetry (into-3080 vs into-5090 rows) |

## Standing verdicts that assumed "no P2P"

| # | Assumption | Where it lives | Which measurement re-checks it |
|---|---|---|---|
| V1 | NCCL path choice: SHM everywhere, NVLS irrelevant (silent-disable), net/socket for cross-rig only | rig-interconnect notes; NCCL env handling in uneven-TP (test_uneven_tp_nccl_env.py); INTEGRATION_R3_VALIDATION "NCCL 2.28.9/SHM (kein P2P)" tables | Q1; re-run the transport check with --baseline against a pre-update capture if one exists, else against the SHM rows recorded in INTEGRATION_R3_VALIDATION |
| V2 | Custom allreduce never activates on this rig (can_p2p false), so its capture/registration path was dead code | custom_all_reduce.py + #195 fix commit (this branch); "Custom-Allreduce blieb inaktiv" in R3 validation | Q5. NOTE: with P2P the path goes LIVE for the first time; the #195 collective-family fix in this branch is the prerequisite for even booting it under solo-draft/lane placements |
| V3 | GDR matrix (#278): no crossover intra-rig; NIC relay serializes | #278 rate tables, /spinning/gdr-uebergabe/ findings | Q3 + Q4; the #278 methodology (parallel-pressure arm) is reproduced in d2d_bench for comparability |
| V4 | #279 dispatcher rate tables (latency terms per link) assume host-staged intra-rig rates | #279 dispatcher config / offload_register latency-term API | Q3; refresh the per-pair rate entries from the new medians (effective values only) |
| V5 | Planner pair matrix (dashboard) ranks pairs on the no-P2P measurements | planner pair matrix / dashboard data | Q3 + Q6; regenerate the pair matrix from d2d_bench JSON |
| V6 | Erg.-7c tier ladder: peer-VRAM parking is not a real tier (transfers would relay through host) | Erg.-7c tier ladder / offload tiering docs | Q2 + Q4: check whether a peer-VRAM park target is now real, and size it with the EFFECTIVE aperture, not the nominal 256 MiB |
| V7 | P2P/NCCL-L0 levers "bring nothing here" (latency hiding and solo-5090 only) | rig-interconnect-P2P memory note | Q1 + Q3; if P2P engages, the lever assessment must be redone from data |
| V8 | Cross-rig RDMA plans (NORDSTERN) treat intra-rig links as SHM-bound floor | NORDSTERN L0/L1 notes | Q3; a higher intra-rig floor changes the cross-rig split, remeasure before re-planning |

## How to fill

1. `bash run_all.sh` (after the update; takes the card locks, writes
   `results/<date>/`).
2. Paste per-row: measurement value, date, verdict `holds | overturned |
   partially`, and follow-up task id if overturned.
3. Rows V1-V8 with `overturned` get their own task before any placement
   code changes -- this file records the delta, it does not authorize
   rewires.
