# #363 — Dynamic regime controller: live per-rank ms/round -> staged plan flips

User idea (2026-07-31): keep speed at the optimum dynamically by watching each
card's per-round milliseconds and moving whichever cut is currently the brake
("everything interlocks with everything").

Verdict: build it — but as a discrete staged controller over planner-solved
plans, not as a continuous chase of the slowest card. The measured record
constrains the design in four ways.

## 1. Actuator reality (what can actually move at runtime)

| lever | runtime-movable | mechanism |
|---|---|---|
| KV token vector | yes, < 1 s delta move | #297 phase-boundary KV resharding |
| VRAM budget per card | yes | #330 VMM page-return dial |
| spec algorithm / k | yes | #156 acceptance-driven ladder |
| lane placement / lending | yes (dual-group runtime) | #274 C2 lend/reclaim |
| weight (MLP/GEMM) shard cut | **no** | no runtime actuator; per-phase boot (#354/#357) |

The controller's action space is therefore the top four rows. A runtime
weight mover is a separate build-out stage; it is also exactly what would
unlock "switch phase arms at the phase boundary" as default behaviour
(the #357 recommendation).

## 2. The brake is often not a card (measured counterexample)

#264 tried the naive version statically: the 3080s clock the prefill, the
5090 waits 390 ms more — shift the cut toward the brake. Result: prefill
+8.2 %, decode -13.7 %, KV capacity -47.9 %, net negative. 69-75 % of the
window is collective-floor cost that no cut reallocation touches. Wait time
on a rank is a symptom, not a lever. The controller must optimise the
end-to-end objective (ms/verify + ms/prefill under capacity constraints),
never a single rank's idle share.

## 3. Build shape: generalise the #287 ladder

The fork already contains the right pattern — the KV-pressure ladder (#287):
N discrete geometry stages, solved and validated by the planner ahead of
time, shadow pre-staging, flip at a stage boundary inside the nesting
family. #363 generalises the trigger from "KV about to burst" to "regime
changed": classify prefill-heavy / decode-heavy / KV-pressure from the
existing per-rank sensing (#252 CollectiveClock, compute vs wait split,
graph-replay-honest), then flip between pre-solved stages via #297/#330.
Runtime never invents a plan; it only selects one.

## 4. Control-loop hygiene (paid-for lessons)

- Noise: throughput follows output content (r = 0.90). Live ms/round must be
  judged against an A-vs-A noise floor; no flip below it.
- Hysteresis and a minimum dwell time per stage, or the controller
  oscillates at regime boundaries.
- Self-conditioning trap (#156 cross-algo bandit): the controller reacts to
  effects it caused itself. Falsifier plan must include an oscillation test
  and a controlled regime-switch replay before any live default.

## Sequencing

Behind the #274 dual-group priority. Phase 1: design doc + falsifier plan.
Phase 2: regime classifier on existing sensing. Phase 3: staged flips via
#297/#330. Phase 4 (separate decision): runtime weight mover.
