# DESIGN_625 — PP/TP optimal phase topology for prefill, draft and decode

Task #625 (with #626 for draft placement). Status: **analysis complete,
measurement pending**. Nothing in this document is implemented yet; the
implementation gate is the measurement in §5.

## 1. Target picture

The model is resident in the decode-TP performance-optimal split — today's
production layout, the measured decode optimum (`--rank-perf-tune
phase-decode`, runbook §4.1.0). Prefill runs in that same TP layout up to a
threshold on the uncached prompt length. Above the threshold the prefill runs
in a PP layout instead, then the server returns to TP for decode. The switch
is made cheap by keeping PP-stage weights DUAL-RESIDENT while free VRAM
allows, with dual residency budgeted as a parkable item that yields under KV
pressure.

## 2. Why PP for prefill (the motivation, and its limit)

68-75 % of the prefill window on this rig is per-layer `all_reduce` cost
(ANALYSE_299, CollectiveClock). The rig has no P2P/NVLink and sits on PCIe
x4/x8/x8, so TP pays that per layer. PP replaces per-layer collectives with
one activation send per stage boundary — structurally, not by tuning.

The limit is the pipeline bubble, and it is what creates the threshold the
target picture asks for. A prompt split into `C` chunks over `P` stages
occupies the pipeline for `C + P - 1` chunk-slots while doing `C` chunks of
work, so the fill/drain efficiency is `C / (C + P - 1)`:

| uncached prompt | chunks at 2048 | PP=3 efficiency |
|---|---|---|
| 2 k | 1 | 0.33 |
| 8 k | 4 | 0.67 |
| 32 k | 16 | 0.89 |

So PP prefill is structurally WORSE than TP at short prompts and better only
once the prompt amortises the bubble. That is the threshold, and its shape is
`C/(C+P-1) x (TP collective saving) > 1`. The crossover point is what §5
measures; it is not derivable from the ladder alone because stage imbalance
and per-stage compute differ on mixed cards.

**This whole argument rests on one unverified assumption**: that successive
chunks of the SAME request actually occupy different microbatch slots and
fill the pipeline. `scheduler_pp_mixin.py:139` (`event_loop_pp`) iterates
`pp_loop_size` slots, each independently calling `get_next_batch_to_run`, and
`scheduler.py:4326` states chunked requests "can start in one microbatch and
end in another microbatch" — consistent with filling, but not proof. The
cross-rig slice-2 counters (runbook §4.9) show "two microbatches in flight
while only one carries a request", which is the OPPOSITE behaviour at low
load. If the pipeline does not fill for a single request, PP prefill is
strictly ~P-times serial and the feature is dead on arrival. **§5 measurement
(a) is therefore a falsification test first and a threshold hunt second.**

## 3. Existing building blocks (what we can stand on)

- **Intra-rig PP boot, proven**: runbook §4.7, `--tp-size 1 --pp-size N`,
  one card per stage, `--pp-layer-ratio` or the score-proportional
  `--pp-stage-ratio` (#201 slice 3). Stage GPU groups must be pairwise
  disjoint (`server_args.py:8669`) — with 3 cards and PP=3 that is satisfied.
- **`--rank-gpu-id` is mandatory under PP on mixed cards** — it forces
  `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1`; without it the sm_86 stage emits
  a Hopper instruction and `ptxas` refuses (runbook §4.7).
- **Full decode CUDA graphs work under PP**; only piecewise graphs are off.
- **#297 KV resharding** (`managers/kv_reshard.py`, `layers/dcp/reshard_plan.py`,
  `POST /kv_reshard`) — the phase-boundary KV mover. Pool addresses stay
  stable across a reshard so decode graphs survive without recapture.
- **#286 short-term offload register** (`model_executor/offload_register.py`,
  `short_term_offload_register.py`) — introduces item classes into the
  EXISTING expert-offload tiering (pinned host pools, H2D wave-in behind
  compute, spill budgets) and owns the policy side. This is the natural home
  for dual-resident PP stage weights.
- **#330 VRAM dial**, **#363 regime controller** (`managers/regime_act.py`,
  which already has a `reshard_arm`), **#388 phase classifier**,
  **#445/#481** PP defect fixes (GGUF sibling-config, world-length rank
  vectors, `expert_stats` per-stage tag).
- **KV pressure ladder** already reserves a rung named `external` for a
  "remote PP stage" warm-standby (`kv_pressure_ladder.py:35`).

## 4. Blockers found in analysis (all verified in code, not from reports)

**B1 — PP and speculative decoding are mutually exclusive at boot.**
`server_args.py:16240-16245` asserts `disable_overlap_schedule and
speculative_algorithm is None` whenever `pp_size > 1`. Hard assert, dies at
arg parse. Registered as `pp_with_spec`, level BLOCKED, in
`planner/rejected.py:342` (its `evidence` line number has drifted; the assert
is now at `:16240` — worth correcting in the register).

Consequence: a server booted `pp_size > 1` has NO MTP/NEXTN decode. The
target picture wants TP+spec decode AND a PP prefill in ONE server, which
this assert forbids for any design that reaches PP by raising boot `pp_size`.

**B2 — no runtime actuator moves weights.** Stated in the runbook §4.1.0 and
confirmed by search: every `reshard` symbol in `srt/` is KV-only. #297 moves
KV tokens, #330 moves the VRAM budget; the MLP weight vector is a boot
decision and "switching arms costs a restart". A TP->PP flip is by definition
a weight-topology change, so it needs a NEW actuator class that does not
exist today. This is the single largest piece of work in #625 and it is what
makes dual residency a PREREQUISITE rather than an optimisation: at
6.4-13 GB/s a >10 GB re-stage is seconds, far above the "small relative to
phase time" bar.

**B3 — dual residency is expensive on this rig.** Weight bytes are the same
under either topology (~27 GB for 27B INT8-W8A8); TP shards each layer, PP
splits by layer. Dual residency therefore costs ~2x weights = ~54 GB against
72 GB total VRAM, before the 8.4 GB standard reserve (3000,2700,2700). That
leaves roughly 10 GB for KV and activations — enough to boot, but it collapses
the KV pool that the decode phase depends on. Dual residency must therefore be
partial and budgeted (park the stage weights that the current phase does not
need), which is exactly the #286 register's semantics.

**B4 — #297 stage-A limits.** KV resharding is weighted-uneven-DCP +
hybrid-linear pool family only, and is refused in combination with PD
disaggregation, hierarchical-cache storage, kv-session-offload,
weightless-KV ranks and the dual-group lane (runbook §4.1.1). The phase
boundary KV move inherits every one of those exclusions.

**B5 — world-MIN `max_total_num_tokens`.** Under PP the token ceiling is
min-reduced across the world group, so the tightest stage sets capacity for
all (runbook §4.7/§4.9, open #201 slice-3 item).

## 4a. MEASURED, 2026-08-07 — the gate is passed, and it moves the design

Window artifacts: `/spinning/gpu-battery-results/2026-08-07_625/`. Both arms
Qwen3.6-27B-INT8-W8A8, ctx 65536, KV fp8_e4m3, `chunked_prefill_size` 2048,
both no-spec, both `--disable-overlap-schedule`, both without hierarchical
cache (see the defect below). Uncached random `input_ids`, `max_new_tokens=1`,
warm-up draw discarded, 3 kept draws. A-vs-A same-boot floor at 8192 tok:
**0.10 % on both arms**.

| uncached tokens | TP=3 median | TP tok/s | PP=3 median | PP tok/s | PP speedup |
|---|---|---|---|---|---|
| 2 048 | 1080.1 ms | 1896.1 | 453.3 ms | 4518.1 | **2.38x** |
| 8 192 | 5475.5 ms | 1496.1 | 1094.8 ms | 7482.9 | **5.00x** |
| 32 768 | 24088.9 ms | 1360.3 | 4611.0 ms | 7106.4 | **5.22x** |

Per-length spreads <= 0.97 %. The differences are 200-500 %, three orders of
magnitude above the floor, so nothing here is close to the noise.

**R1 is resolved positively: the pipeline DOES fill.** PP throughput RISES
from 4518 tok/s at one chunk to 7483 tok/s at four — that rise is only
possible if successive chunks of the same request occupy different microbatch
slots. The §2 falsification arm did not trigger.

**Correctness, not just speed.** The PP arm answers 14*3=42 and Jupiter
identically to the TP arm, and retrieves a needle planted mid-prompt across a
6862-token (3.4-chunk) pipelined prefill. Multi-chunk PP prefill is correct.

**Decode confirms the phase split.** PP decode bs=1 measured ~31 tok/s against
the documented TP+NEXTN 112 tok/s (runbook §4.1.0, INT8 decode arm) — PP is
~3.6x WORSE at decode. Prefill and decode really do want opposite topologies,
which is the premise of #625 and is now measured on both sides.

### What the numbers change about the design

**There is no threshold.** PP wins at every length measured, including 2048
tokens — one chunk, where PP has maximum bubble and literally zero pipelining
(stage 0's 32 layers, then stage 1's 16, then stage 2's 16, strictly serial).
It still wins 2.38x there, because on this rig the per-layer collectives cost
more than the entire serialisation penalty. That is ANALYSE_299's 68-75 %
figure showing up from the other side.

So the §1 target picture's central mechanism — a prompt-length threshold that
flips TP->PP and back — **is the wrong shape for this rig**. A per-request
flip would also have to pay a weight-topology switch (B2) that no actuator
performs. The measurement instead points at Route A (§7) in its strongest
form: **two persistent groups, no per-request switch at all.** A PP prefill
group and a TP+NEXTN decode group both stay resident; the router sends prefill
to the PP group always (not above a threshold), and the only per-request cost
is the KV handover, which #212 measured at 1.8 % of TTFT. The "threshold"
degenerates into "is the handover cheaper than 2.38x of prefill", which at
2048 tokens it already is.

The honest caveat: the TP arm ran the plain VRAM-auto split. Every
concentrated phase-prefill candidate was refused UNBOOTABLE at reserve
4500,2700,2700 (rank 1 residual below the derived 4016 MiB demand), so the
planner CHOSE "keep plain VRAM-auto split". That is the production-RESIDENT
decode layout, which is the right baseline for the target picture; per runbook
§4.1.0 the ideal INT8 prefill vector would have added only ~6.1 %, which does
not touch a 2.38-5.22x gap.

### Defect found: PP + disk HiCache wedges at warm-up

`--enable-hierarchical-cache` with the file backend under `--pp-size 3` never
becomes ready. Health stays 503 indefinitely; the launcher sits in
`_wait_and_warmup`. Two identical py-spy samples, i.e. wedged and not merely
slow: **PP2 blocked in `isend`** (`send_tensor_dict` ->
`_pp_send_output_to_next_stage`) while **PP0 spins in `_drain_async_work` ->
`check_hicache_events`** inside `_get_new_batch_prefill_raw`, so PP0 never
posts the matching recv. The same topology without hicache boots in ~2 min.
It fails SILENTLY (no refusal, no error line), which is the AUDIT_505 shape.
Evidence: `pp3_hicache_wedge_pyspy.txt`, `pp3_hicache_wedge.boot.log`.
Not fixed here; it needs its own task, and until then PP boots must refuse
hierarchical cache loudly at parse time rather than hang.

## 5. Measurement plan (the implementation gate)

Two boots, both no-spec — per the register, a PP number must never be placed
next to a NEXTN number, so the TP arm is booted no-spec too.

- **(a) prefill ladder**: PP=3 (3 stages x TP=1, one card per stage,
  `--pp-stage-ratio` score-proportional) vs TP=3 with
  `--rank-perf-tune phase-prefill`. Uncached prompt ladder 2 k / 8 k / 32 k.
  ms-per-round canon, same-boot A-vs-A floor FIRST with the warm-up draw
  discarded and draws back-to-back; work-matched counters.
  Falsification arm: if PP=3 prefill is at or below single-stage throughput,
  the pipeline is not filling (§2) and #625 stops here with that finding.
- **(b) switch cost**: control path + KV flip with pre-staged stage weights,
  against a full weight copy, measured separately. Feeds the threshold.
- **(c)** draft placement per phase topology (#626) — only if (a) wins.

Threshold formula shape: flip to PP when
`C/(C+P-1) x T_tp_prefill(C) - T_pp_prefill(C) > T_switch_in + T_switch_out`,
with `T_switch_*` taken from (b) and `C` the UNCACHED chunk count (prefix-cache
hits do not pay prefill and must not count toward the threshold).

## 6. Implementation sketch (only if §5 (a) wins)

Opt-in flag, default OFF, default path byte-identical; everything gated, no
interleaving into the default path. Given B1 and B2 the honest sequencing is:

1. A weight-topology actuator: stage-weight staging into the #286 register as
   a new parkable item class, reusing the expert-offload movement machinery
   rather than building a second one.
2. A PP prefill execution path that does not require boot `pp_size > 1`
   (B1) — i.e. an internal stage-partitioned forward, not the engine's
   `pp_group` machinery.
3. Phase-boundary KV reshard from the PP write layout into the TP decode
   layout via #297, inheriting B4's exclusions as loud refusals.

Refuse loudly, at parse time, for: GGUF checkpoints where the declared-depth
probe cannot resolve (the #481 sibling-config path), any spec algorithm if the
chosen mechanism reaches boot `pp_size` (B1), and every #297 stage-A exclusion
(B4). No silent fallbacks.

## 7. Route pricing (phase 1.5): PD composition vs a new actuator

Two ways to reach "TP+NEXTN decode AND PP prefill in one serving system".

### Route A — PD disaggregation (#99/#106/#107/#212, #111 NCCL-L0 transport)

A separate prefill group with its own topology (boots no-spec, may boot PP)
feeding the resident TP+NEXTN decode group over the existing KV handover.
The threshold becomes a ROUTER decision: short prompts prefill locally in the
decode group's TP layout, long prompts go to the PP prefill group.

**The engine already has this shape.** `scheduler.py:6612` dispatches
`DisaggregationMode.PREFILL` with `pp_size > 1` to
`event_loop_pp_disagg_prefill` (`scheduler_pp_mixin.py:248`), which is
`event_loop_pp` plus bootstrap/release steps for the KV transfer. So a PP
prefill group inside a PD pair is a supported configuration, not a new one.

**Measured handover cost (#212, runbook §4.8, cross-rig over 40G):** of a
2.892 s satellite TTFT, ~53 ms was the 98 MiB KV transfer (1.8 %) and ~136 ms
handshake plus scheduling. The remaining 2.72 s was the satellite's own slow
compute (2080 Ti at 2385 tok/s against the 5090's 10850) — **that term does
not apply intra-rig**, which is what makes the route worth pricing here. The
runbook's own conclusion: "the transport is not what stands in the way".

**But the route is NOT available today, and the coordinator's premise needs
one correction.** `arg_groups/pd_disaggregation_hook.py:34-46` silently
disables speculative decoding on BOTH PD arms:

> speculative decoding is not supported in disaggregation mode on this fork —
> the MTP/EAGLE draft KV pool is uneven-head-sharded (not DCP token-sharded),
> so its transfer would need general uneven head reslicing.

It is an auto-disable with a warning, not a hard error ("design ruling", so
shared launch configs keep working). A PD decode arm launched with
`--speculative-algorithm NEXTN` therefore comes up with spec silently OFF —
which is exactly the failure shape AUDIT_505 is about, and it would quietly
destroy the decode optimum the whole feature exists to protect.

**Why this is still much cheaper than Route B.** The named obstacle is narrow:
it is the DRAFT KV pool at the boundary, not the main KV and not the weights.
NEXTN is a single draft layer. Two ways out, both bounded:
(i) recompute the draft layer's prompt KV locally on the decode arm from the
transferred hidden states — one layer against the main model's full depth; or
(ii) the general uneven head reslicing the comment asks for.
Either is a far smaller build than moving ~27 GB of weights between two live
layouts.

### Route B — in-process weight-topology actuator (the original §6 sketch)

Requires B2 (a weight mover that does not exist) and runs into B1 unless the
PP path avoids boot `pp_size`. Dual residency at ~54 GB of 72 GB (B3) is a
standing tax on the KV pool.

### Verdict

**Route A is the preferred route, conditional on measurement (a).** It turns
#625 from "build a new actuator class" into "compose existing groups +
threshold routing policy + residency budgeting", and it dissolves B1 and B2
as blockers — B1 because prefill and decode are then separate servers with
separate worlds (the assert only ever constrained ONE world), and B2 because
each group's weights are loaded once at its own boot instead of being moved.
What it does NOT dissolve is B3 (two groups' weights are still two copies,
now across two processes, so the #286/#330 residency budgeting is unchanged)
and it adds one new must-fix: the silent spec auto-disable above.

Route A also inherits the #212 operational traps, several of which are real
here: the PD handshake compares only `page_size` and `kv_cache_dtype` (two
arms with different weights pair and produce fluent nonsense — the preflight
script is mandatory), and
`--disaggregation-decode-enable-radix-cache` is a hard ValueError for
Mamba/SSM models, which Qwen3.6-27B is.

## 7a. Route A guards (#631a/#631b), and the slice-2 variant choice

### Guard 1 — model identity at the PD handshake

`try_ensure_parallel_info` (`disaggregation/common/conn.py:411-454`, sole call
site `decode.py:788`) compares exactly two fields: `page_size` and
`kv_cache_dtype`. Nothing else. Two arms holding DIFFERENT WEIGHTS pair
happily and produce fluent nonsense (#212), which is the worst failure mode
available to us because the output looks like output.

**The guard is half-built already, and that decides the design.** The fork's
own NCCL transport carries `TransportIdentity.model_identity_hash`
(`disaggregation/nccl/contract.py:146`), a sha256 over model path, revision,
dtype, quantization and kv_cache_dtype, and it explicitly REUSES
`compute_model_identity_hash` from `mem_cache.hicache_storage` rather than
inventing a hash. So the correct slice is not "add a checksum" but "lift the
existing identity into the transport-independent handshake": add the hash to
`PrefillServerInfo` (`common/conn.py:67-90`, which today carries only topology
fields) and compare it in `try_ensure_parallel_info` beside the two existing
checks, with the same refusal shape. One hash function, three transports, one
place to compare.

Refusal must name both hashes and the likely cause (different checkpoints, or
the same checkpoint at different revisions/quantizations), because the operator
symptom — plausible but wrong text — gives no hint on its own.

### Guard 2 — decode-arm radix cache on a hybrid model

`--disaggregation-decode-enable-radix-cache` raises for SSM models at
`mem_cache/kv_cache_builder.py:195-199` (`is_hybrid_ssm`). Qwen3.6-27B is a
hybrid GDN model, so this is live for us. The existing PD hook
(`pd_disaggregation_hook.py:80-96`) already refuses this flag against
`--enable-hisparse`, the `fake` backend and speculation — but NOT against
hybrid SSM, so that one fires late, at cache-build time, rather than at parse.

The guard is to move it earlier, next to its three siblings, so the decode
arm's recipe cannot carry the flag past argument parsing. The model-type read
needed for it is not new: the #481 sibling-config canon
(`declared_config_path`) already resolves a checkpoint's declared config at
parse time for `--pp-stage-ratio`, and must be reused rather than
re-implemented — a second config reader is how the GGUF `.gguf`-file trap got
in last time.

### Slice 2 (#631b): I am NOT taking variant (i), and here is the number

The steer's default was (i) recompute the draft layer's prompt KV locally on
the decode arm. The concrete reason it loses, on this model's real geometry
(`hidden_size` 5120, `num_key_value_heads` 4, `head_dim` 256, KV fp8):

| what crosses the boundary | bytes per token | at 32 768 tokens |
|---|---|---|
| (i) last-layer hidden states, bf16 | 5120 x 2 = 10 240 B | **335 MB** |
| (ii)/(iii) one draft layer's K+V, fp8 | 2 x 4 x 256 = 2 048 B | **67 MB** |

Variant (i) is **5x more traffic**, and worse, it is a NEW payload: the PD
handover moves KV, and hidden states for every prompt position are not
something it carries today. The premise that (i) is "more local" does not
survive contact with the requirement — the decode arm cannot reconstruct the
draft layer's input from what it receives. It gets the main model's K and V,
and K/V are projections; they do not invert back into hidden states. So "one
layer of recompute" silently implies "ship 335 MB of activations first".

**What I propose instead — variant (iii), a canonical-layout draft KV.** The
PREFILL arm computes the draft layer's KV during its own prefill, where the
hidden states already exist in flight and the marginal cost is one layer on
top of 64. It ships that KV in a canonical FULL-HEAD layout, and the decode
arm slices out the heads its own sharding owns. This gets (ii)'s 67 MB without
building (ii)'s general uneven head reslicing: with 4 KV heads over 3 decode
ranks nothing divides evenly, which is exactly the generality the code comment
balks at — and a canonical intermediate layout removes the need for it, since
neither side ever has to understand the other's sharding.

Cost of (iii): the prefill arm must hold the draft layer's weights (one layer),
and one canonical layout has to be specified and versioned. That is a smaller
and more local change than either (i)'s new activation payload or (ii)'s
general resharder.

I am flagging this as a DEVIATION from the steer and will not build it until
it is confirmed, since the steer named (i) explicitly.

### Hard precondition, not mine to fix

#630 (PP x disk-HiCache warm-up wedge, §4a) blocks SHIPPING a PP prefill
group at all: disk HiCache is mandatory in every serving boot, and the two
together wedge silently. Slices 1 and 2 are both reachable without touching
it, but a PP prefill group cannot go into production until #630 closes.

## 8. Open risks

- R1: pipeline may not fill for a single request (§2) — kills the feature.
  Measured first, deliberately.
- R2: B2 means #625 is a large build, not a wiring job. The measurement must
  justify it before any of §6 starts.
- R3: dual residency vs KV pool (B3) may make the flip unfundable at the 27B
  point even if PP prefill wins in isolation.
- R4: `planner/rejected.py:342` `pp_with_spec` evidence line had drifted from
  the code (cited `:11214`, real site `:16240-16245`); corrected in this
  branch under the register's own re-check rule.
- R5 (Route A): the PD spec auto-disable is SILENT. Any #625 work on Route A
  must convert it into a loud refusal for the decode arm before it can be
  trusted, or a boot that looks correct will have lost NEXTN without saying
  so.
