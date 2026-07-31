# ANALYSE #347 — cross-feature optimizations ("everything feeds everything")

Survey of optimizations that exploit crossings between existing subsystems,
checked against the discard register (nothing re-proposed from it). Ordered
by expected gain. Items 1, 2+3, 4 are registered as tasks; 5 folds into
#344/#333-M3/#341; 6 is a later planner objective.

## 1. Idle workbench — generalize the #341 tenant (task #347)

Training is one idle tenant among many. The same machinery (idle detection,
preemption via checkpoint-and-release, ledger lease) carries a queue of
useful idle work:

- TRT staircase rung compilation (#337) — rungs build themselves overnight.
- The 3 missing sm120 FP8 tuner shapes (#255 remnant) — already specified
  as an idle-tuner protocol, becomes just another queue entry.
- CUDA-graph prewarm for registered-but-cold engines (cuts promotion
  latency on the residence ladder).
- Cold-tier compression of hibernate images (#306).
- Self-benchmarking: the dashboard's "absent" factor tiles re-measure
  themselves when the rig is idle.

One scheduler, one priority order, everything preemptible by serving
demand. #341-M1 builds the tenant runtime; this generalization is the
follow-up slice: the work-queue abstraction plus the first two non-training
tenants.

## 2. Cross-class pairing on dedicated engines (task #348a)

NVDEC/NVENC are dedicated silicon: video decode/encode does not contend
for SMs. A K3 video job can therefore co-run with K1 LLM decode nearly
free — but "nearly" must be measured (memory-bandwidth and PCIe share).
Slice D's pairing policy (SM-saturating vs not) currently knows only K1
lanes; extend the objective to K1xK3 co-tenancy. Measurement first:
ms/Verify with and without a concurrent video job, per card, against the
noise floor.

## 3. One cost model, many consumers (task #348b)

Three placement planners price the same physics with separate code and
assumptions: the key solver (K1 splits), video shard_plan (M2 chunks), and
the upcoming expert placement (#302). Unify on one cost library: per-card
rate x pair-matrix hop cost. Every measurement (card short-probe, comm
suite) then improves all consumers at once, and each new class planner is
a consumer, not a re-implementation. Guard: the #216/#264 lesson — the
roofline never ranks splits; measured points do.

## 4. Integration boot matrix as a standing bug net (task #349)

Cross-feature bugs are invisible to git and to single-feature tests
(#132 x weightless NCCL hang; the #340 arm matrix carrying a silent
SGLANG_UNEVEN_DCP=1). A small, time-boxed boot matrix over feature
crossings (spec x DCP x offload x dual-lane x video co-tenancy), each arm
printing its EFFECTIVE configuration (the dcp_report.sh pattern) and gated
on coherence/byte checks, catches this class systematically. Runs as an
idle-workbench tenant (see 1) once that exists; until then, a manual
pre-release sweep.

## 5. Preview taps as a lane principle (folds into #344 / #333-M3 / #341)

The video live-watch directive generalizes: diffusion decodes a low-res
latent preview every N steps (cheap); training streams loss curves over
the same events channel; LLM already streams tokens. One observation
pattern across classes, one frontend building block. No separate task —
carried as a design rule in the three feature tasks.

## 6. Energy as a selectable objective (later)

The energy harness measures J/token and the power-limit sweep machinery
exists. Exposing max-throughput vs max-efficiency as a planner objective
(J/token, J/frame) is a portable dial — it re-derives from measurement on
any rig. Queued behind the current performance line; no task yet.

## Deliberately not proposed

- Weight-sharing between engine instances via IPC: in the discard register.
- Model cascades / difficulty routing: changes outputs — lossy class,
  behind the attention-quant line per the gain-first order.
- Anything that only helps this specific rig (rig-is-lower-bound rule).
