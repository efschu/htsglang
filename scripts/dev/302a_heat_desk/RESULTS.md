# #302a desk falsification — raw results

Desk only. No GPU, no boot, no model. Everything below is computed by the two
scripts in this directory from `expert_stats_*.json` artifacts recorded by the
#390 instrument on four independent boots. Reproduce with:

```
python3 simulate_heat.py  /spinning/gpu-battery-results/2026-08-02_439_arm3/expert_stats_equal.tp{0,1,2}ep0.json
python3 transfer_heat.py  --train <A.json> --test <B.json> [--test ...]
python3 transfer_heat.py  --lookahead <run.json>
```

## 0. What the artifacts contain, and what they do not

Per rank, per layer: `num_experts`, `resident_count`, whole-run
`expert_activations[e]` (activation count per local expert id), hit/miss
tallies, `peakedness`, and the offload's own `residency` fetch/H2D tally.

They do **not** contain a per-token routing trace. Two consequences, stated
before any number below:

* A genuine intra-run **windowed** simulation (re-rank every N decode steps
  from the traffic of the preceding N steps) is **not computable** from these
  files. §2 substitutes the harsher cross-run transfer test instead of
  pretending otherwise.
* "Does layer N's top-k predict layer N+1's top-k **for the same token**" is
  likewise not answerable. §3 answers the aggregate form of the question and
  says which form it answered.

## 1. Sanity gate + oracle ceiling (`simulate_heat.py`)

Dataset: `2026-08-02_439_arm3/expert_stats_equal.tp{0,1,2}ep0.json`
(86 086 tokens, 77 959 forwards, DSV4-Flash, uneven TP=3, base plan
`30407,19080,19080`).

The static layout is reconstructed from `plan_load_time_staging`'s rule (the
#82 pad expert at `E-1` pinned to the lowest slot, remaining slots filled in
ascending id order). Reconstruction vs the recorded number:

| rank | E / R | recorded | reconstructed | delta | oracle (same R) | lift |
|---|---|---|---|---|---|---|
| tp0ep0 | 115 / 56 | 0.7623 | 0.7623 | **0.0000** | 0.9836 | **+22.12 pp** |
| tp1ep0 |  72 / 31 | 0.8427 | 0.8427 | **0.0000** | 0.9844 | **+14.18 pp** |
| tp2ep0 |  72 / 31 | 0.8463 | 0.8463 | **0.0000** | 0.9850 | **+13.87 pp** |

The delta is exactly zero on all three ranks, to the last activation. The
simulation is therefore measuring the real placement, not a model of it.

**Excluding the pad expert** — it absorbs every foreign-id token under the #82
expert-dim shard and is a structural always-hit, so it flatters the headline
figure — the placement quality of the static layout is far worse than 0.76-0.85
suggests:

| rank | static (pad excluded) | oracle (pad excluded) | lift |
|---|---|---|---|
| tp0ep0 | 0.4799 | 0.9641 | +48.42 pp |
| tp1ep0 | 0.4318 | 0.9438 | +51.19 pp |
| tp2ep0 | 0.4226 | 0.9437 | +52.11 pp |

Read plainly: with the pad's contribution removed, the load-time resident set
catches slightly under half of the routed mass, and a same-sized set chosen by
heat catches ~95 %. The static choice is close to arbitrary with respect to
what the router does.

Per-layer, the lift is not uniform — the worst layers on tp0 are
L40 (+40.7 pp), L36 (+35.1), L31 (+31.7), L22 (+30.3); the best are the early
layers L0/L1 (+12.4/+13.1). Full per-layer tables are printed by the script.

## 2. Does a heat ranking GENERALISE? (`transfer_heat.py`)

The oracle above ranks on the very run it is scored on and is an upper bound no
online policy can reach. The achievable question is what a ranking learned from
PAST traffic is worth on FUTURE traffic. Four independent boots with the same
geometry are available:

| run | tokens | note |
|---|---|---|
| `2026-08-01_417_dsv4arch/expert_stats_w5` | 125 904 | different day |
| `2026-08-02_439_arm3/expert_stats_equal` | 86 086 | the briefing's dataset |
| `2026-08-02_394_linkshards/expert_stats_equal` | 661 985 | longest |
| `2026-08-02_desync_graph_proof/expert_stats_eager` | 29 326 | shortest |

Train on one, score on another. `cap %` is the fraction of that pair's oracle
ceiling the transferred ranking actually captured. Diagonal entries are the
oracle by construction and are shown for scale.

### rank tp0ep0 (E=115, R=56)

| train \ test | 417_w5 | 439_arm3 | 394_link | graph_eager |
|---|---|---|---|---|
| **417_w5** | 0.9944 (100 %) | 0.8862 (+12.39 pp, 56.0 %) | 0.8435 (+7.98, 40.7 %) | 0.9051 (+12.91, 60.1 %) |
| **439_arm3** | 0.8259 (+4.64, 21.6 %) | 0.9836 (100 %) | 0.8937 (+12.99, 66.3 %) | 0.9456 (+16.96, 79.0 %) |
| **394_link** | 0.8206 (+4.11, 19.1 %) | 0.9449 (+18.26, 82.5 %) | 0.9599 (100 %) | 0.9280 (+15.20, 70.8 %) |
| **graph_eager** | 0.8369 (+5.75, 26.7 %) | 0.9377 (+17.54, 79.3 %) | 0.8766 (+11.28, 57.5 %) | 0.9906 (100 %) |

static baseline: 0.7794 / 0.7623 / 0.7638 / 0.7760 respectively.

### rank tp1ep0 (E=72, R=31)

| train \ test | 417_w5 | 439_arm3 | 394_link | graph_eager |
|---|---|---|---|---|
| **417_w5** | 0.9944 (100 %) | 0.9139 (+7.13, 50.3 %) | 0.8891 (+5.30, 41.1 %) | 0.9142 (+8.22, 51.6 %) |
| **439_arm3** | 0.8607 (+2.08, 13.5 %) | 0.9844 (100 %) | 0.9243 (+8.82, 68.3 %) | 0.9577 (+12.57, 78.9 %) |
| **394_link** | 0.8522 (+1.24, 8.0 %) | 0.9556 (+11.30, 79.7 %) | 0.9653 (100 %) | 0.9421 (+11.01, 69.1 %) |
| **graph_eager** | 0.8668 (+2.69, 17.4 %) | 0.9503 (+10.76, 75.9 %) | 0.9163 (+8.02, 62.1 %) | 0.9912 (100 %) |

static baseline: 0.8398 / 0.8427 / 0.8361 / 0.8320.

### rank tp2ep0 (E=72, R=31)

| train \ test | 417_w5 | 439_arm3 | 394_link | graph_eager |
|---|---|---|---|---|
| **417_w5** | 0.9946 (100 %) | 0.9115 (+6.52, 47.0 %) | 0.8881 (+5.13, 40.2 %) | 0.9200 (+7.70, 51.5 %) |
| **439_arm3** | 0.8464 (+1.93, 11.5 %) | 0.9850 (100 %) | 0.9250 (+8.82, 69.1 %) | 0.9592 (+11.63, 77.7 %) |
| **394_link** | 0.8413 (+1.42, 8.5 %) | 0.9568 (+11.05, 79.6 %) | 0.9645 (100 %) | 0.9474 (+10.45, 69.8 %) |
| **graph_eager** | 0.8592 (+3.20, 19.1 %) | 0.9476 (+10.13, 73.0 %) | 0.9143 (+7.76, 60.7 %) | 0.9926 (100 %) |

static baseline: 0.8272 / 0.8463 / 0.8367 / 0.8430.

### Reading

* **Every off-diagonal cell is positive.** The smallest transferred lift in the
  whole matrix is **+1.24 pp** and the largest is **+18.26 pp**; the median is
  around +8 pp. Nothing here is close to the 2-3 pp "weak cell" threshold.
* **Within the 2026-08-02 family** (three boots, same day, similar workloads)
  transfer captures **57-83 %** of the oracle ceiling.
* **The 2026-08-01 run is the odd one out**: rankings transfer INTO it poorly
  (8-27 %) while it transfers OUT of it acceptably (40-60 %). Its workload
  differs from the 08-02 family. That asymmetry is the argument FOR a live
  re-rank rather than a shipped static profile: the useful ranking is the one
  learned from the traffic you are actually serving.
* These are the **limit case of staleness** — a whole different boot, day and
  workload. A re-rank every N decode steps inside one session sees traffic much
  closer to what it is about to be scored on, so 57-83 % is a floor for the
  in-session case, not a ceiling.

## 3. #302-lookahead: does layer N's heat predict layer N+1's? (bonus)

**Answered form**: aggregate, not per-token (see §0). Per adjacent layer pair,
over the layer's whole-run activation vector with the #82 pad expert excluded
(it is the top expert in every layer and would manufacture correlation).

| run / rank | pairs | Spearman (mean / min / p50 / max) | top-R overlap (mean) | chance | top-8 overlap (mean) | chance |
|---|---|---|---|---|---|---|
| 439_arm3 tp0 | 42 | -0.0199 / -0.2200 / -0.0147 / 0.1505 | 0.4847 | 0.487 | 0.0595 | 0.0702 |
| 439_arm3 tp1 | 42 | -0.0221 / -0.3029 / -0.0194 / 0.1648 | 0.4409 | 0.431 | 0.0893 | 0.1127 |
| 439_arm3 tp2 | 42 | +0.0117 / -0.1979 / +0.0037 / 0.1867 | 0.4555 | 0.431 | 0.1012 | 0.1127 |
| 394_link tp0 | 42 | -0.0284 / -0.2540 / -0.0068 / 0.1809 | 0.4741 | 0.487 | 0.0655 | 0.0702 |

**Verdict: no exploitable signal.** Rank correlation between adjacent layers'
heat is indistinguishable from zero (|mean Spearman| <= 0.03, and the sign is
not even consistent across ranks). Top-R set overlap sits on the chance line
`R/E` to within 2.5 pp. Top-8 overlap is at or slightly BELOW chance once the
pad expert is removed — the 0.17-0.20 figure a naive computation produces is
entirely the pad expert appearing in both top-8 lists.

Consequences worth recording, because they are the useful part of a negative
result:

* A cross-layer prefetch hint ("layer N routed to X, so pre-stage X's
  neighbours for layer N+1") has nothing to work with at this grain.
* Conversely, **one shared heat ranking cannot serve several layers**: each
  layer's hot set is its own. #302a is per-layer for a measured reason, not a
  conservative default.
* This does not refute a per-token cross-layer correlation, which these
  artifacts cannot see. Measuring that needs `SGLANG_MOE_OFFLOAD_TRACE`
  (the per-layer routed-id log that already exists) and is a separate,
  cheap-to-collect desk item.

## 4. Verdict

**MATERIAL — proceed to build.** The oracle ceiling is +13.9 to +22.1 pp of
activation-grain hit rate at unchanged resident-set size, and a ranking learned
under maximal staleness still banks +5 to +13 pp of it. The miss fraction on
tp0 goes from 0.238 to as little as 0.054 at the ceiling and to ~0.11 at the
measured transfer rate — the term `ANALYSE_456` §2.2 names as the dominant one
in decode latency, roughly halved.

The lookahead sub-cell is **REFUTED at this grain** and is reported as such.

## 5. What this is NOT

* Not a decode-throughput claim. Hit rate is a necessary condition for the H2D
  reduction, not a measured tok/s. The A/B in `AB_SPEC.md` is what would make a
  throughput claim, and it has not run: **BOOT-PENDING**.
* Not a windowed simulation. See §0.
* Not a claim about any other model or checkpoint. Every number is
  DSV4-Flash on this rig's uneven TP=3 geometry.
