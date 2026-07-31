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

## Correction and standing order (2026-07-31, user)

Two corrections to §2 of this note, from the user, both accepted:

1. **#264 falsified the STATIC compromise layout, not phase switching.** The
   user's proposal is to run each phase in its own optimum and move the
   layout between phases — #354's table already quantifies that this
   captures BOTH columns (+22.6 % prefill AND the full 447 tok/s decode),
   which no static layout does. §2's warning applies only to a controller
   that picks one layout for all regimes.
2. **The 5090's prefill wait IS chronic under-utilisation** (390 ms more
   wait, #252) — fixed by concentration, now planner-self-solved (#357).
   Only the DECODE wait is collective-latency floor where concentration
   measurably hurts (-4.6..-20.8 %, #354). Both are measured; the switcher
   collects both optima.

Standing order (user, restated 2026-07-31): ALL model kinds must be
cuttable (uneven TP/DCP), dynamically by load, spillable, with CUDA graphs
— always, wherever not hard-blocked. TensorRT engine rungs the same
(cuttable, load-dynamic, spillable) wherever they make sense (#337).

## Status matrix against that order

**Cuttable (uneven TP/DCP)** — done for: dense FP8/BF16/INT8-W8A8/AWQ/GPTQ
(alignment sibling family closed), GGUF dense + MoE, NVFP4 (dequant lane),
122B MoE offload, Gemma4 SWA-hybrid (incl. SWA-DCP), Llama family, video
multi-card (pull-queue), diffusion uneven-SP (measured-driven since #348b).
Open: Mistral (#130), AR-audio/Omni (#333/#334), draft-KV DCP (#108,
user-deferred), GPTQ-MoE load crash in repair (#283, running).

**Dynamic by load (runtime movers)** — built: KV token vector (#297,
kv_reshard.py), VRAM budget (#330, vram_dial.py), KV-pressure plan flips
(#287, kv_pressure_runtime.py — the first runtime geometry flip), spec
algo/k (#156/#75), lane lending (#274 C2), session spill (FCFS + budget +
async tick). NOT built: the weight-shard flip — see below. #363 assembles
these into the regime controller.

**Weight cut at runtime: not blocked, deliberately not built yet.** Reasons,
in order: (a) the planner chain had to solve stages correctly first
(#265→#298→#324→#353→#357 — a mover that moves onto wrongly-solved plans is
worse than none); (b) the cheap movers covered the decode side, phase boots
the prefill side; (c) #274's context-overlay/lane substrate is where a
mover belongs — built earlier means built twice; (d) for MoE the mover
largely EXISTS (expert offload moves weights live: prefetch, presplit,
waves; #302 dynamic placement is the registered extension). Everything
needed exists in pieces: live weight movement (experts), VMM remap (#93),
fast serialize/restore (#89 hibernate), group handover (#261), lane weight
sharing by data_ptr identity (#274). Physics of a flip on this rig: moving
the MLP delta between concentrated and auto layouts is GBs over PCIe with
the 5090 on x4 — order 3-6 s including quantized repack and re-capture
from pre-staged pools (#102/#286). Hence: staged flips on sustained regime
shifts, queue-aware (a large admitted prompt predicts the prefill burst —
flip proactively), never per tick. Per-tick rebalancing stays hard-blocked
by move-cost >> tick-duration.

**Spillable** — built: KV host-tier + budget + latency classes + async
tick; expert spill tier (RAM-resident cold experts; harder-quant tier #126
open); graph capture pools + IO buffers taggable/offloadable (#102);
short-term offload register in progress (#286: graph rungs, drafter, lane
workspaces, cold lane); hibernate-to-disk (#89; MoE image deferred);
suspend-to-RAM. Named hard rule: GDN state stays resident for session
spill (KV-only spill), a real constraint, not a gap.

**CUDA graphs** — the standing default everywhere; graph-safety closed
across GGUF/uneven-DCP/MTP-bs>1/expert-offload/weightless/PD/barlink-device.
Named exceptions with reasons: bar1ep EP dispatch forces eager (host
collective for token counts, by design — #361); hybrid-GDN standard-GQA
check can drop the prefill graph. Stage flips require pre-captured pools
per stage — exactly what #102/#286 make holdable/offloadable.

**TensorRT rungs (#337)** — became real today: TRT 11.2.1.2 installed
(user-approved), first two production engines built (SR fp16, sm86+sm120,
parity 45.5/48.1 dB > 40 gate, build discipline recorded: I/O signature,
dynamic-shape profiles, consumer matrix, per-arch build), ORT-EP
incompatibility (so.10 vs so.11) documented with direct-TRT fallback.
The #337 program itself — precompiled rungs, granular, offloadable
(VRAM/RAM/disk, local/remote, hot/cold±compression), graphs-vs-TRT
crossover per regime, sm86 honesty — is queued, not started. Its rungs
slot into the same #363 staircase: a TRT engine is one more stage a regime
flip can select, and spilling an engine is file movement (engines are
per-arch artifacts on disk by construction).
