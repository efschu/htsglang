# Fixtures for the work-matched counter rule (#523, rule from #482)

All four directories descend from ONE real measurement window:
`/spinning/gpu-battery-results/2026-08-03_439_green` (the #439 green-corridor
re-proof, DeepSeek-V4-Flash UD-IQ3_XXS, TP=3, `--rank-moe-ratio link`). Nothing
here is invented traffic; what is derived is derived by an arithmetic that is
written down below, so a reader can redo it.

The dumps are trimmed: the real `expert_stats_*.json` carries a 43-entry
`layers` array (~72 KiB each) that `read_arm.py` never opens. Only `totals` is
kept, verbatim unless a line below says otherwise.

## `green_final/` — REAL, unmodified

The window's FINAL (post-teardown) dump revision, `totals` byte for byte. This
is the pair the published numbers are quoted from:

| rank | equal h2d | compute h2d |
|---|---:|---:|
| tp0 | 1878.6 GiB | 1871.4 GiB |
| tp1 | 1197.4 GiB | 706.8 GiB |
| tp2 | 1244.1 GiB | 1643.3 GiB |

Work point: `equal` 163486 tokens, `compute` 163572 — 0.053 % apart, which is
what "work-matched" means in practice. At the measured links
14.42 / 6.45 / 13.41 GB/s this gives the transfer term 199.33 s → 139.35 s.

## `green_preteardown/` — real bytes, derived work counters

The same window's PRE-TEARDOWN revision. Its H2D bytes are real: they are the
per-rank GiB the window's own `read_equal.txt` / `read_compute.txt` printed
(one decimal, converted back with `* 2**30`):

    equal   1821.0 / 1158.0 / 1203.6 GiB
    compute 1722.7 /  648.4 / 1512.2 GiB

Those two files predate the work counters, so the counters here are DERIVED:
each rank's counters are its final counters scaled by that rank's own share of
its final H2D bytes (0.9693 / 0.9671 / 0.9674 on `equal`, 0.9205 / 0.9174 /
0.9202 on `compute`). That is precisely what "each rank writes its dump on its
own 45 s timer" produces, and it is why this revision refuses as
`non-final-revision` before any cross-arm question is even asked.

## `window_gap/` — real bytes, one work point per arm

The same real pre-teardown bytes, but with a single work point per arm rather
than one per rank: `equal` at 96.8 % of its run, `compute` at 91.9 % of its
own — the two fractions the green window measured and recorded in
`ARM3_COMPUTE.md`, "Which revision to read". This isolates the CROSS-arm defect
from the intra-arm one: each arm looks internally final, and only the
comparison is invalid. It is the case the pre-#523 tooling passed silently, and
with the gate disarmed it still reproduces the published-and-withdrawn
**1.5028x** exactly.

## `legacy_nowork/` — final bytes, work counters removed

`green_final` with `tokens` / `forwards` / `activations` deleted: a dump
revision from before the counters existed. A ratio taken from it is not merely
wrong, it is unfalsifiable — there is nothing to check the work point against.
Refuses as `missing-counter`.
