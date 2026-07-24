# Benchmark charts

Measured on the reference rig (1× RTX 5090 + 2× RTX 3080, TP=3 uneven-DCP,
Qwen3.6-27B-FP8). Numbers are medians of clean (non-degenerate) decodes;
GPU non-determinism gives ±15–20% run-to-run spread, so single-run claims
are avoided.

## `draft_crossover_tokps.svg`
Draft-usage crossover at context 4096 on a coding workload. The context gate
switches the drafter at the 4096 policy threshold: 100% DFLASH below, 100%
NEXTN above (confirmed, single switch per run at ctx 4104–4126). tok/s per
config, below vs above 4096, with speculative accept-length annotated.
Takeaway: the cross-algo switching-worker overhead (~13–15%) nearly eats
DFLASH's acceptance lead, so cross-algo ≈ pure NEXTN on code; only static
DFLASH wins on code (but loses on prose and above 4096); adaptive-k beats
fixed k=3.

## `spill_tokens_over_time.svg`
KV-session-offload (spill): cumulative generated tokens vs wall-clock for a
co-resident run. A session whose KV is spilled to host RAM keeps decoding
in place over host-streamed KV at the slower floor slope, while the
device-resident sessions run at full speed — graceful degradation instead
of preemption.
