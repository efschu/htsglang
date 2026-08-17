# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Auto-performance uneven-TP split mode (``--rank-tp-ratio auto-performance``).

Builds on the measured M22 feasibility findings (see ROADMAP "Performance-
oriented uneven split"): decode throughput is FLAT (+-2%) across every
representable split on heterogeneous rigs, so the attention/GDN/KV split
STAYS the VRAM-auto split. The mode's stage-1 levers are:

  (a) the fine-grained dense-MLP unit vector (--rank-mlp-ratio family plan):
      family-selective concentration of MLP units toward compute-strong
      ranks is a measured strict prefill/throughput win (M22 C6:
      auto + 5,1,1 = +10% prefill, +7% concurrent, ~0 context) because the
      INT4 MLP bytes move while attention/KV-head and GDN/SSM splits (the
      context-expensive families) stay put;
  (b) TP-degree reduction ("drop the slow-linked card", +55-76% prefill at
      -72% context per M22) -- emitted as a RECOMMENDATION LOG only, never
      applied silently.

SSM/GDN shifting is deliberately NOT a lever: the mamba state pool moves
with the GDN units (~4.7 MiB/req/unit) and collapses context (M22 C3).

Concentration is bounded by the decode-knee guard: no rank's share of the
streamed weight bytes may exceed its share of the rig's EFFECTIVE decode
bandwidth (measured M23: beyond that knee decode regresses (-4.8% at 16,1,2)
while prefill gains saturate at the knee-exact C6 level). Effective is not
streaming peak: it is the probe's decode-shaped GEMV rate, with a residual
exponent for the part a dense probe cannot see -- see
``PerfCostModel.decode_bw_basis``. Every candidate also
carries a predicted decode-step delta in the log, accepted or rejected: the
guard is a model, and a reader has to be able to disagree with it.

The MLP vector is derived from a MEASURED hardware profile (stage-0
micro-probe: per-card GEMM + memory-bandwidth score, pairwise NCCL link
matrix), cached under ~/.cache/sglang keyed by (sorted GPU UUIDs, driver
version) so the probe runs once per rig. The GEMM score is taken in the
checkpoint's OWN weight format, per card and per lane -- a dense bf16 number
reports the wrong compute RATIO for a quantized checkpoint, which is what the
prefill objective is made of (#296); see the "GEMM lanes" block below. The
chosen vector is printed as a
PINNABLE hint (same UX as SGLANG_UNEVEN_TOKEN_VECTOR): passing
``--rank-tp-ratio auto --rank-mlp-ratio <vector>`` reproduces the split with
no probe and no optimizer. SGLANG_PERF_REPROBE=1 forces a re-probe.

Context floor (``--rank-perf-loose-ctx-percent X``): every candidate's max
context is PREDICTED with the same per-rank capacity math the pool sizing
uses (budgets -> weight/mamba/reserve terms -> per-rank token capacity P_r
-> C = min_r(P_r/ratio_r) * sum(ratios), continuous optimum min(sum P,
64*min P)); only candidates with C >= (100-X)% of the VRAM-auto split's
prediction are admissible. X=0 (default) admits only free gains -- which
exist, because MLP-only shifts conserve the summed free bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: 2: per-GPU membw_read_gbs / membw_copy_gbs / membw_gemv_gbs added. A v1
#: profile carries no GEMV rate, so the decode divisor falls back to the
#: streaming peak with a named reason rather than silently.
#: 3: per-GPU gemm_lanes / gemm_lane_notes added -- the GEMM probe measured in
#: the QUANTIZED formats a checkpoint actually runs in, not only dense bf16.
#: A v2 profile carries no lane, so an fp8 plan on one falls back to bf16 with
#: a named warning rather than silently scoring the ranks in the wrong format.
#:
#: Both bumps only ADDED fields, so an older cached profile is migrated and
#: topped up, never discarded -- see ``_PROFILE_VERSION_FIELDS``.
PROFILE_VERSION = 3
PROFILE_CACHE_DIR = os.path.expanduser("~/.cache/sglang")

#: Probe GROUPS: the per-GPU keys one measurement pass produces. The stage-0
#: probe can run any subset of them, which is what makes a version bump
#: affordable -- see ``_PROFILE_VERSION_FIELDS``.
PROBE_GROUP_GEMM = "gemm"
PROBE_GROUP_LANES = "lanes"
PROBE_GROUP_MEMBW = "membw"
_PROBE_GROUP_FIELDS: Dict[str, Tuple[str, ...]] = {
    PROBE_GROUP_GEMM: ("gemm_tflops",),
    PROBE_GROUP_LANES: ("gemm_lanes", "gemm_lane_notes"),
    PROBE_GROUP_MEMBW: (
        "membw_gbs",
        "membw_read_gbs",
        "membw_copy_gbs",
        "membw_gemv_gbs",
    ),
}
_FIELD_PROBE_GROUP: Dict[str, str] = {
    field: group for group, fields in _PROBE_GROUP_FIELDS.items() for field in fields
}

#: Per-GPU keys each PROFILE_VERSION ADDED. Both bumps so far were purely
#: additive: every value the previous version measured still means the same
#: thing. So a cached profile of an older version is MIGRATED, not discarded --
#: its values are carried over and only the fields the newer versions added are
#: marked missing and re-measured, in a lazy top-up that runs the probe GROUPS
#: those fields belong to and nothing else.
#:
#: This is the general rule for future bumps, not a one-off for 2->3. Throwing
#: a whole profile away over an added field costs a full stage-0 probe on the
#: next boot -- including the pairwise link matrix, which is the slowest and by
#: far the most failure-prone phase (task #303: an unreachable rendezvous there
#: charged 600 s to every auto-performance boot). Declare the added fields
#: here, register the probe group that measures them above, and the bump costs
#: only that group.
#:
#: A bump that CHANGES the meaning of an existing field is not a migration and
#: must not be listed here: drop the entry from the cache key instead, so the
#: old value cannot be carried forward.
_PROFILE_VERSION_FIELDS: Dict[int, Tuple[str, ...]] = {
    2: ("membw_read_gbs", "membw_copy_gbs", "membw_gemv_gbs"),
    3: ("gemm_lanes", "gemm_lane_notes"),
}

#: Top-level key holding the REASON for anything the profile does not carry --
#: the same mechanic as the per-lane ``gemm_lane_notes`` (#298a): an absent
#: measurement stores why it is absent, so a consumer reports a named gap
#: instead of silently substituting a number.
PROFILE_NOTES_KEY = "notes"

#: The two classes a reason recorded through ``_profile_note`` can carry
#: (task #310). The #303 mechanic ("an absent measurement stores why") was
#: built for facts about the CARD -- a lane that genuinely cannot run on that
#: architecture -- and every existing note is one of those. But a reason can
#: also describe the PROBING INTERPRETER rather than the hardware (missing
#: sgl_kernel, a mock fallback standing in for it, ...): that is not evidence
#: about the card and must never be written into a cache file that is keyed
#: by GPU UUID and read back on every future boot, on any interpreter.
#:
#: ``NOTE_CLASS_CARD`` (the default, and the only class every note used
#: before this task) is persisted exactly as before. ``NOTE_CLASS_ENVIRONMENT``
#: is logged and then DROPPED -- ``_profile_note`` does not write it into the
#: profile at all.
NOTE_CLASS_CARD = "card_note"
NOTE_CLASS_ENVIRONMENT = "environment_error"

#: Probe shapes (model-relevant): GEMM = one chunked-prefill MLP matmul
#: (2048 tokens x 5120 x 17408 bf16), MEMBW = a decode-style weight-streaming
#: GEMV over a ~1.3 GB bf16 matrix (the lm_head-class shape of m20).
_PROBE_GEMM_M, _PROBE_GEMM_K, _PROBE_GEMM_N = 2048, 5120, 17408
_PROBE_GEMV_ROWS, _PROBE_GEMV_K = 131072, 5120
#: Warmup / timed iterations shared by every GEMM lane below. One setting for
#: all of them on purpose: the prefill objective reads a RATIO between lanes
#: and between cards, and a ratio between two differently-timed measurements
#: is not one.
_PROBE_GEMM_WARMUP, _PROBE_GEMM_ITERS = 10, 60
#: Weight-scale block of the fp8 lanes, ``[block_n, block_k]``. Block-scaled
#: e4m3 is what the fp8 checkpoints on this fork's rigs ship (Qwen3.6-27B-FP8:
#: ``weight_block_size [128, 128]``), and the block size is not free: it sets
#: the scale traffic the Marlin and dequant lanes pay per output tile. Both
#: probe dimensions divide it exactly, so the lanes measure the aligned path.
_PROBE_FP8_BLOCK = (128, 128)

# ---------------------------------------------------------------------------
# Capacity-prediction constants (calibrated against the M22/M23 boot logs of
# this fork's uneven-TP pipeline; see HANDOFF M22/M23).
# ---------------------------------------------------------------------------
#: Per-rank overhead (CUDA context share inside the budget, NCCL buffers,
#: attention workspaces, allocator slack) subtracted from the byte budget
#: before converting to tokens. Uniform across ranks; only candidate-relative
#: and rank-relative fidelity matters for the floor decision.
_PREDICT_OVERHEAD_MIB = 1280
#: The auto-mamba activation reserve (MAMBA_AUTO_ACTIVATION_RESERVE_MIB).
_PREDICT_MAMBA_ACT_RESERVE_MIB = 1024
#: EXTRA budget charged to the solo draft HOST only
#: (--speculative-draft-placement solo), on top of the draft weights and the
#: globally-sized draft KV pool that the families / KV cell already cover:
#: the draft's own attention workspace (flashinfer scratch, roughly fixed) and
#: its decode CUDA graphs (which scale with the captured batch sizes, hence
#: with max_running_requests). Calibrated on the reference rig, where the host
#: allocated 634 MiB past its budget at max_running_requests=2 and 2266 MiB at
#: 4; the values below bound both with margin, because under-reserving here
#: OOMs the card mid-decode while over-reserving only costs some KV.
_SOLO_HOST_WORKSPACE_MIB = 512
_SOLO_HOST_GRAPH_MIB_PER_REQ = 512
#: Minimum viable per-rank token capacity: below this the weighted-DCP owner
#: rule degenerates (a rank must own >= 1 of every virtual block; M22 C3b
#: measured the resulting context collapse).
_PREDICT_MIN_RANK_TOKENS = 4096
#: Token-vector granularity of the weighted-DCP converged optimum.
_PREDICT_TOKEN_UNITS = 64
#: GEMM efficiency assumed when converting probe TFLOPS into per-token
#: prefill compute time (cancels in candidate ratios; kept for logging).
_PREDICT_GEMM_EFF = 0.6
#: Exponent of the link-bandwidth penalty folded into a rank's prefill
#: score: a compute-strong card behind a narrow link attracts fewer units.
_PREDICT_LINK_ALPHA = 0.25
#: Decode-knee guard tolerance: a candidate may not raise any rank's share
#: of the total streamed weight bytes beyond that rank's share of the rig's
#: total memory bandwidth (times 1+tol). Decode is bandwidth-bound and
#: measured FLAT below this point (M20/M22); beyond it the strong card
#: becomes the decode lockstep bottleneck AND the extra prefill gain does
#: not materialize (M23 measured: 16,1,2 at 56.5% bytes-share on a 51.9%
#: membw-share card = decode -4.8%, prefill +10.0% -- the SAME +10% the
#: knee-exact C6 vector 5,1,1 delivers at decode +0.8%). The guard is the
#: trust region of the prefill model as much as a decode protection.
_PREDICT_DECODE_KNEE_TOL = 0.02
#: MLP unit grids finer than this land close to the continuous decode knee,
#: so the strict byte-share test alone is trustworthy. Coarser grids (FP8
#: dense ~136 units, GGUF K-quant ~68 units) can only realize the split in
#: large steps, and the nearest representable vector can sit measurably ABOVE
#: the knee even when its whole-model streamed-byte share still reads under
#: the bandwidth share (M27d: FP8 4,1,1 = 51.5% bytes-share < 51.9% membw
#: share, yet decode -14/-24% vs the auto split -- the per-layer lockstep
#: bottleneck bites before the whole-model share crosses). For such coarse
#: grids we require extra headroom below the knee (see below).
_PREDICT_KNEE_COARSE_UNITS = 256
#: On a coarse MLP grid, require this many unit-steps of streamed-byte-share
#: headroom below a rank's bandwidth share before admitting a candidate, so
#: the optimizer rounds DOWN to the last safe vector instead of the lumpy
#: overshoot. Calibrated on the M27d rig (5090 + 2x3080, membw share 51.9%):
#: this rejects the FP8 4,1,1 knee overshoot (picking the 3,1,1 class) while
#: the fine AWQ 544-unit grid is unaffected and keeps its measured-win 5,1,1.
#:
#: What this stands in for, and what would replace it. The guard compares
#: WHOLE-MODEL streamed-byte shares, i.e. it models the decode round as
#: max_r(total bytes_r / bw_r). A TP decode step does not work that way: it
#: all-reduces after every mixer block and every MLP block, so the round time
#: is the SUM over sync points of the per-sync MAX, and the two are only equal
#: when the same rank is the bottleneck everywhere. They are not equal here --
#: attention/GDN follow the base plan while the MLP follows the candidate
#: vector -- and max-of-sums is the smaller of the two, which is why the guard
#: needs a hand-set margin to catch what it structurally understates. Summing
#: per-sync maxima instead is parameter-free and moves the three #216
#: predictions from -7.7/-5.8/-1.8 to +1.1/+4.3/+8.3 against measured
#: +6.0/+13.4/+15.5, i.e. it fixes the SIGN with no exponent at all. It is not
#: adopted here: it is a second change to the same model, it needs its own
#: campaign (per-block testing rejects 3,1,1 too, which does cost +6.0 %), and
#: bundling it would leave neither change with clean evidence.
_PREDICT_KNEE_COARSE_HEADROOM_UNITS = 2
#: The decode roofline's divisor is a rank's ACHIEVED weight-read bandwidth,
#: not the card's streaming peak. A card 2.32x faster on a streaming benchmark
#: is not 2.32x faster at pulling a weight shard through the GEMV kernels a
#: bs=1 decode step is made of, and on a mixed rig the two fractions do not
#: cancel: the decode advantage of the 5090 over a 3080 that the measured
#: step times imply here is 1.46x (1.70x on the three #216 points alone,
#: 1.46x once the #264 point is in the fit -- see the refit note below).
#: Reading the peak ratio as an achieved ratio puts the ceiling for the
#: fast rank far too high, and the guard then waves through exactly the
#: concentration that makes that rank the lockstep pacer.
#:
#: Two things supply the correction, in this order:
#:
#:  1. MEASUREMENT. The probe's decode-shaped GEMV reads the same buffer the
#:     streaming kernels read, and reaches less: 1532 vs 1663 GB/s on the
#:     5090, 717.4 vs 717.8 on the 3080 -- the fast card gives up 8 % of its
#:     stream to the GEMV, the slow card gives up nothing. Ratio 2.32 -> 2.14.
#:     That is a fifth of the way (22 % in log terms) to the implied 1.46,
#:     and it costs no constant: it is read off the rig.
#:  2. The residual exponent below, for the rest.
#:
#: Exponent applied to the achieved GEMV rate: ``bw_eff ~ bw_gemv ** BETA``.
#: Fitted (task #216 follow-up) on ms per SPECULATIVE STEP over four MLP
#: vectors, interleaved cold boots, KV ownership vector pinned, radix cache
#: defeated on the prefill axis: base [63,37,36] 30.10 ms, 3,1,1 31.90,
#: 4,1,1 34.12, 6,1,1 34.76 -- rms 0.38 ms, boot-to-boot floor 0.6-0.8 %.
#: BETA = 1 predicts -8 to -2 % for those vectors where +6 to +16 % was
#: measured.
#:
#: REFIT #265, on the #264 A/B. That campaign added a fourth point on a
#: different base plan: base ``2,1,1`` (units [68,34,34]) -> ``6,1,1``
#: (units [102,17,17]), ms/verify 32.599 -> 37.307 = +14.4 %. At the previous
#: 0.70/0.28 pair the model reads that step as +8.7 % (the plan log's
#: base-relative +6.4 % against the VRAM-auto split, which is where the
#: reported "2.2x too mild" comes from -- the A/B's arm A was the pinned
#: 2,1,1, not the base). The miss is structural in the pacer, not in the
#: size of a percentage: at 0.70 the model puts the DECODE PACER on a 3080
#: at both campaign bases, which makes mild concentration a predicted GAIN
#: (-2.1 % for 2,1,1) -- while every measured concentration from the base is
#: a COST. At or below 0.50 the 5090 paces at every vector on both campaign
#: bases, the predicted-gain region disappears, and the four points fit
#: together.
#:
#: Joint (BETA, nonweight-fraction) grid over the four points, rms in
#: percentage points:
#:     0.70 / 0.28 (shipped)  +7.1 / +11.2 / +16.2 / +8.7   rms 3.13
#:     0.50 / 0.35 (this)     +8.0 / +11.8 / +16.4 / +14.7  rms 1.37
#: against measured +6.0 / +13.4 / +15.5 / +14.4. The rms optimum is a flat
#: valley (BETA 0.50-0.57 x fraction 0.36-0.37, rms 1.338-1.342), so the four
#: points still do NOT separate the two scalars -- they pin the pair, exactly
#: as the three did. BETA is taken at 0.50 rather than at the rms minimum
#: because rms is flat across the whole range while the PACER is not: the
#: 0.52-0.57 end sits on the flip, where 2,1,1 still reads as a small gain,
#: and 0.50 is inside the plateau where the sign is right on both bases. Cost
#: of that choice: 0.007 rms points.
#:
#: What the fourth point adds is the SIGN of the curve near the base, which no
#: value of the fraction can supply: the fraction multiplies the weight-term
#: ratio, so it can damp a predicted cost but never turn a predicted gain into
#: one.
#:
#: What the residual IS. The probe reads dense bf16; a decode step reads
#: QUANTIZED weights through dequantising kernels -- native FP8 on sm_120,
#: Marlin upconvert on sm_86, MMVQ for GGUF -- and those lanes do not sit at
#: the same fraction of DRAM peak as a dense read. A dense probe cannot
#: measure that, and a synthetic stand-in does not help: at per-layer
#: granularity the same measurement swings 1.98..2.58 on the row count alone
#: (see _bench_membw_rates), which is the size of the effect being chased.
#: Closing this the rest of the way means timing the ACTUAL quantized kernels
#: per lane, which is a per-lane, per-checkpoint measurement, not a constant.
#:
#: The PREFILL side now does exactly that (see the GEMM lanes below), and the
#: decode side deliberately does not follow it here. Two reasons, both about
#: evidence rather than effort. The lanes measure a GEMM at M=2048; a bs=1
#: decode step reads weights at M=1, where the same kernels sit at a different
#: fraction of DRAM peak and the dequant lane's per-forward expansion is
#: amortized over one token instead of a block (measured on this rig: forcing
#: the dequant fallback costs 8.1 % of prefill and 69.9 % of decode). And this
#: exponent was FITTED against the dense GEMV divisor on four measured points;
#: swapping the divisor without refitting would keep the number and change
#: what it means. A decode-shaped quantized probe plus a refit is the way to
#: close it, and it is its own campaign.
#:
#: Sample width: ONE rig (5090 + 2x 3080, PCIe, no P2P) and one checkpoint
#: family (FP8 dense, sm_120 native lane against sm_86 Marlin upconvert).
#: The direction -- achieved ratios are compressed relative to probed ones --
#: is general; this particular value is not claimed to be.
_PREDICT_DECODE_GEMV_RESIDUAL = 0.50
#: FALLBACK exponent, applied to the STREAMING peak when no usable GEMV rate
#: is available (profile from before PROFILE_VERSION 2, or the saturation
#: check below rejecting the measurement). Same fit, same four points, same
#: caveat -- but it starts from a divisor that has measured none of the
#: compression, so it has to supply all of it. Its existence is why the
#: fallback is NAMED and logged rather than taken silently: 0.45 and 0.50 are
#: not interchangeable, they belong to different divisors. Held at the
#: achieved ratio the GEMV path produces on the reference rig
#: (2.131 ** 0.50 = 1.46 = 2.319 ** 0.45), so a rig that falls back does not
#: silently change WHICH decode ratio the guard believes, only how much of it
#: was measured.
_PREDICT_DECODE_BW_COMPRESSION = 0.45
#: Saturation floor for the GEMV probe, as a fraction of the same card's
#: streaming rate. Below this the GEMV is not bandwidth-bound at all (a
#: kernel that fell back to something pathological on an architecture we have
#: not seen), and its rate says nothing about weight streaming. Measured
#: here: 0.92 on the 5090, 1.00 on the 3080 -- so the floor is set where only
#: a broken kernel can reach it, NOT where a legitimately slower decode lane
#: would. A guard that fires on a real measurement would put us back where
#: the max() was.
_PROBE_GEMV_SATURATION_FLOOR = 0.25
#: Ceiling, likewise as a fraction of the streaming rate: a GEMV that reads
#: FASTER than a pure stream is being served from cache, not DRAM, and is not
#: a bandwidth measurement either. Not hypothetical -- a 64 MiB matrix is L2
#: -resident on a 5090 (96 MiB L2) and reads at 2.2 TB/s, past its own DRAM
#: peak. The 1.34 GB probe matrix is far past any L2 on any card we know of,
#: so this is a guard against future probe shapes and unfamiliar cards.
_PROBE_GEMV_CACHE_CEILING = 1.05
#: Fraction of a bs=1 decode step that is NOT weight streaming (attention
#: over the KV, the speculative draft and verify passes, collectives, kernel
#: launch). The weight-term ratio alone overstates the step-time delta,
#: because this part of the step does not move when the MLP split does;
#: folding it in is what takes the reported cost from "the right sign" to
#: "within ~2 points of measured".
#:
#: Not independently fitted: this value and the exponent above trade off
#: along a valley, so the data pin their combination, not either one. The
#: fourth point (#264, refit #265) narrows the valley without closing it --
#: BETA 0.50-0.57 x f 0.36-0.37 all land at rms 1.34 -- so 0.35 is the round
#: value inside it, at rms 1.36. Separating the two does not need a wider
#: campaign, it needs a DIRECT measurement: profile one decode step and split
#: device time between weight-reading kernels and everything else. Then f is
#: read off, and the exponent is what the step times alone determine. Note it
#: is not a constant even in principle -- attention time grows with context --
#: though the #216 pair at ctx 400 vs 12000 moved the 6,1,1 cost only from
#: +16.7 % to +16.4 %, so over that range it is nearly flat.
_PREDICT_DECODE_NONWEIGHT_FRACTION = 0.35
#: Fraction of a prefill step that does NOT move with the weight split.
#:
#: The prefill model books ONLY shard-proportional time: per-token GEMM
#: FLOPs over the probe's GEMM rate, plus the per-layer all-reduce. That is
#: a model of a step whose whole duration scales with the rank's shard --
#: true for the large fused matmuls themselves, not for the step around
#: them. The measured #216 campaign (and the boots it ran in) had
#: ``prefill.backend='disabled'``: no captured prefill graph, so every
#: layer's kernels are launched eagerly, and that launch/dispatch overhead
#: -- plus the non-GEMM ops (norms, GDN scan, attention softmax, radix
#: bookkeeping) -- is split-INVARIANT: it does not shrink when MLP units
#: leave a rank. Omitting it inflated every predicted concentration gain:
#: predicted +9.2/+15.1/+21.6 % for 3,1,1 / 4,1,1 / 6,1,1 against measured
#: slope gains +6.4/+9.0/+13.0 % (over-prediction x1.4-1.7 with the current
#: probe inputs; the campaign-time inputs gave 23.7 % vs 13.0 %, x1.8).
#: A fresh graphless anchor points the same way: 27B-FP8, TP=4, ctx ~6995,
#: ``prefill.backend='disabled'`` measured 146.5 tok/s prefill (server log;
#: 159.4 client) -- far below any shard-proportional reading of the probe.
#:
#: The model therefore charges an invariant term, anchored on the base
#: plan:  t(v) = t_sharded(v) + f/(1-f) * t_sharded(base),  which leaves
#: candidate RANKING untouched (the term is constant across candidates) and
#: deflates the reported gains. Fitted on the three measured #216 slope
#: gains: at 0.35 the predictions land at +5.8/+9.3/+13.0 % (rms 0.4
#: points). Refit recipe on other hardware: measure the prefill slope
#: (unique random input ids per request, ``#cached-token: 0`` proven) for
#: the base split and one concentration vector, then solve
#:   f = (r_model - r_meas) / (r_meas * (r_model - 1))
#: with r = t_base/t_cand from ``prefill_time_model`` at f=0 and from the
#: measurement. Sample width: ONE rig, one checkpoint family, eager
#: (graphless) prefill; a graph-captured prefill path has a smaller
#: invariant share, so refit rather than reuse when that lands.
_PREDICT_PREFILL_INVARIANT_FRACTION = 0.35
#: TP-degree recommendation: a GPU whose best pairwise link is below this
#: fraction of the rig's best link is called out as a drop candidate.
_TP_DROP_LINK_FRACTION = 0.7
#: ... provided the remaining ranks' budgets still fit the weights with
#: this fill factor of headroom.
_TP_DROP_FIT_FACTOR = 0.85


@dataclasses.dataclass(frozen=True)
class PerfCalibration:
    """The cost model's fitted/assumed scalars, gathered into one seam.

    What is MEASURED, per rig, on every machine: the stage-0 probe's GEMM
    rate, streaming bandwidth and decode-shaped GEMV rate per card, the NCCL
    link matrix, and the NVML totals. Those numbers never come from here.

    What is NOT measured on the machine the model runs on -- the four scalars
    below. Each was fitted (or bounded) on the reference rig (RTX 5090 +
    2x RTX 3080, PCIe without P2P, Qwen3.6-27B-FP8, uneven TP=3) and is a
    HYPOTHESIS on any other machine. This class exists so that refitting them
    elsewhere is a parameter change, not a code edit: pass an instance to
    ``PerfCostModel`` or set the matching ``SGLANG_PERF_*`` env var.

    Field -> shipped value -> what pins it -> how to refit:

    ``decode_gemv_residual_exp`` (0.50, ``_PREDICT_DECODE_GEMV_RESIDUAL``)
        Exponent on the measured GEMV rate. Pinned by four measured ms-per-
        speculative-step points (#216 follow-up + the #264 A/B). Refit:
        interleaved cold boots of the base split plus one concentration
        vector, solve ``beta = ln(achieved decode ratio) / ln(gemv ratio)``.
    ``decode_peak_compression_exp`` (0.45, ``_PREDICT_DECODE_BW_COMPRESSION``)
        Fallback exponent on the STREAMING peak when no usable GEMV rate
        exists. Same fit, same points, larger residual (the divisor measured
        none of the compression).
    ``decode_nonweight_fraction`` (0.35, ``_PREDICT_DECODE_NONWEIGHT_FRACTION``)
        Share of a bs=1 decode step that is not weight streaming. NOT
        independently identified -- it and the exponent trade off along a
        valley; separating them needs a profiled decode step.
    ``prefill_invariant_fraction`` (0.35, ``_PREDICT_PREFILL_INVARIANT_FRACTION``)
        Share of a prefill step that does not move with the weight split
        (eager launch overhead, non-GEMM ops). Fitted on the three measured
        #216 prefill slope gains; refit formula at the constant's definition.

    A field left at None reads the module-level constant at call time, so a
    test that patches the constant and a refit that sets the env var both
    take effect without fighting each other.
    """

    decode_gemv_residual_exp: Optional[float] = None
    decode_peak_compression_exp: Optional[float] = None
    decode_nonweight_fraction: Optional[float] = None
    prefill_invariant_fraction: Optional[float] = None

    @classmethod
    def from_env(cls) -> PerfCalibration:
        """Overrides from the ``SGLANG_PERF_*`` refit seam (environ.py);
        unset vars stay None and fall through to the shipped constants."""
        from sglang.srt.environ import envs

        return cls(
            decode_gemv_residual_exp=envs.SGLANG_PERF_DECODE_GEMV_RESIDUAL_EXP.get(),
            decode_peak_compression_exp=(
                envs.SGLANG_PERF_DECODE_PEAK_COMPRESSION_EXP.get()
            ),
            decode_nonweight_fraction=(
                envs.SGLANG_PERF_DECODE_NONWEIGHT_FRACTION.get()
            ),
            prefill_invariant_fraction=(
                envs.SGLANG_PERF_PREFILL_INVARIANT_FRACTION.get()
            ),
        )

    # -- resolved values (explicit override > shipped reference-rig fit) ----

    @property
    def gemv_residual(self) -> float:
        v = self.decode_gemv_residual_exp
        return _PREDICT_DECODE_GEMV_RESIDUAL if v is None else float(v)

    @property
    def peak_compression(self) -> float:
        v = self.decode_peak_compression_exp
        return _PREDICT_DECODE_BW_COMPRESSION if v is None else float(v)

    @property
    def nonweight_fraction(self) -> float:
        v = self.decode_nonweight_fraction
        return _PREDICT_DECODE_NONWEIGHT_FRACTION if v is None else float(v)

    @property
    def prefill_invariant(self) -> float:
        v = self.prefill_invariant_fraction
        f = _PREDICT_PREFILL_INVARIANT_FRACTION if v is None else float(v)
        # A fraction >= 1 has no reading as "share of the step"; clamp so a
        # bad refit value degrades to a loud, finite prediction, not a
        # division by zero.
        return min(max(f, 0.0), 0.95)

    def overridden_fields(self) -> List[str]:
        """Names of the fields explicitly set (for the plan log: a refit in
        effect must be visible, or a foreign value silently poses as the
        reference fit)."""
        return [
            f.name
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        ]

    def borrowed_fields(self) -> List[str]:
        """Names of the fields still at the REFERENCE-RIG fit (#434).

        The mirror of :meth:`overridden_fields`, and the more consequential
        half on any machine that is not the rig these were fitted on: they
        are scalars from one rig, one checkpoint family and one prefill mode,
        and a plan that reports only the OVERRIDDEN case makes silence mean
        "reference fit" without saying so. Each constant's own definition
        above carries the campaign that produced it and the recipe for a
        refit; this method is what puts the fact in the boot's log.
        """
        return [
            f.name
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is None
        ]


# ---------------------------------------------------------------------------
# Stage 0: hardware micro-probe (runs in a SUBPROCESS so the launcher stays
# free of CUDA state; a few seconds per rig, cached afterwards).
# ---------------------------------------------------------------------------


def _nvml_gpu_inventory() -> Tuple[List[dict], str]:
    """Per-CUDA-device {uuid, name, total_mib} in CUDA enumeration order,
    plus the driver version.

    Card identity comes from the #331 identity map (#397), which keys on the
    NVML UUID and bridges CUDA ordinal <-> NVML index over the PCI BDF. This
    used to bridge through ``server_args._torch_to_nvml_gpu_index_mapping``
    and fill any gap with ``mapping.get(cuda_idx, cuda_idx)`` -- the identity
    substitution that names the wrong card on exactly the rigs the bridge
    exists for. The whole rig now resolves or the probe refuses: this
    inventory keys the hardware profile cache, so one misattributed card
    would persist a profile that describes a different machine.
    """
    import pynvml
    import torch

    from sglang.srt.registry import nvml as registry_nvml

    imap = registry_nvml.identity_map(allow_cuda_init=True)
    placed = sorted(
        (c for c in imap if c.cuda_ordinal is not None),
        key=lambda c: c.cuda_ordinal,
    )
    # The probe describes the cards CUDA hands it, so a card CUDA can see and
    # the map cannot place is the refusal case -- NVML cards masked out of
    # this process are not (they are none of the probe's business).
    visible = torch.cuda.device_count()
    if len(placed) != visible:
        raise registry_nvml.DeviceOrderUnresolvedError(
            f"the hardware micro-probe inventory: torch sees {visible} CUDA "
            f"device(s) but the identity map placed {len(placed)}, so at "
            "least one card cannot be named. Refusing to guess: this "
            "inventory keys the hardware profile cache, and one misattributed "
            f"card persists a profile of a different machine (#397). "
            f"Reason: {registry_nvml.cuda_bridge_diagnosis()}. Cards "
            "present:\n" + (imap.describe() or "  (no NVML devices)")
        )

    pynvml.nvmlInit()
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
    finally:
        pynvml.nvmlShutdown()

    gpus = [
        {
            "cuda_index": card.cuda_ordinal,
            "uuid": card.uuid,
            "name": card.name,
            "total_mib": card.total_mib,
        }
        for card in placed
    ]
    return gpus, driver


def profile_cache_path(
    uuids: Sequence[str], driver: str, version: Optional[int] = None
) -> str:
    """Cache path of the profile for this (GPU set, driver, schema version).

    ``version`` defaults to the current ``PROFILE_VERSION``; passing an older
    one addresses the file a previous release wrote, which is what the
    migration reads."""
    key = json.dumps(
        [sorted(uuids), driver, PROFILE_VERSION if version is None else version]
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return os.path.join(PROFILE_CACHE_DIR, f"hw_profile-{digest}.json")


def legacy_profile_paths(uuids: Sequence[str], driver: str) -> List[Tuple[int, str]]:
    """``(version, path)`` of every OLDER schema version's cache file for this
    rig, newest first. The schema version is part of the cache key, so an
    older profile lives at a different path and has to be looked up
    explicitly."""
    return [
        (v, profile_cache_path(uuids, driver, version=v))
        for v in range(PROFILE_VERSION - 1, 0, -1)
    ]


def _profile_fields_through(version: int) -> List[str]:
    """Every per-GPU field declared by ``_PROFILE_VERSION_FIELDS`` up to and
    including ``version``."""
    fields: List[str] = []
    for v in sorted(_PROFILE_VERSION_FIELDS):
        if v <= version:
            fields.extend(_PROFILE_VERSION_FIELDS[v])
    return fields


def migrate_profile(profile: dict) -> Tuple[dict, Dict[str, List[str]]]:
    """Lift a cached profile of an older ``PROFILE_VERSION`` to the current one.

    Returns ``(migrated, {uuid: [missing field, ...]})``. Every value the old
    profile carries -- per-card rates, the link matrix, the probe timestamp --
    is carried over unchanged; only the fields the newer versions ADDED are
    reported missing, for the caller to top up lazily. The returned profile is
    a deep copy; the input is not modified."""
    migrated = json.loads(json.dumps(profile))
    old_version = int(migrated.get("version") or 0)
    migrated["version"] = PROFILE_VERSION
    migrated["migrated_from"] = old_version
    wanted = _profile_fields_through(PROFILE_VERSION)
    missing: Dict[str, List[str]] = {}
    for uuid, entry in (migrated.get("gpus") or {}).items():
        gaps = [f for f in wanted if f not in entry]
        if gaps:
            missing[uuid] = gaps
    return migrated, missing


def probe_groups_for_fields(fields: Sequence[str]) -> List[str]:
    """The probe groups that have to run to measure ``fields`` -- in the fixed
    order the full probe runs them, so a top-up and a full probe measure a
    card in the same sequence (a GEMM pass leaves the card in a different
    clock state than a bandwidth pass, and the order is part of the
    measurement)."""
    groups = {_FIELD_PROBE_GROUP[f] for f in fields if f in _FIELD_PROBE_GROUP}
    return [
        g
        for g in (PROBE_GROUP_GEMM, PROBE_GROUP_LANES, PROBE_GROUP_MEMBW)
        if g in groups
    ]


def _profile_note(
    profile: dict, key: str, reason: str, note_class: str = NOTE_CLASS_CARD
) -> None:
    """Record WHY ``key`` is absent from ``profile`` -- unless ``reason`` is a
    ``NOTE_CLASS_ENVIRONMENT`` fact, which is logged and NEVER persisted (see
    the class docstrings above ``NOTE_CLASS_CARD``): a missing dependency in
    the interpreter that happened to run the probe says nothing about the
    card and must not survive into the cached file for the next reader."""
    if note_class == NOTE_CLASS_ENVIRONMENT:
        logger.warning(
            "auto-performance probe: %s (environment fact, not recorded in "
            "the profile -- %s is a per-card cache keyed by GPU UUID)",
            reason,
            key,
        )
        return
    profile.setdefault(PROFILE_NOTES_KEY, {})[key] = reason


# ---------------------------------------------------------------------------
# GEMM lanes: the matmul a checkpoint's weights actually run through.
#
# A quantized checkpoint does not compute at the card's dense bf16 rate, and
# the gap is card-specific. The prefill objective consumes a compute RATIO
# between ranks, so a probe in the wrong format does not just shift the scale
# -- it reports the wrong ratio. Measured on the reference rig (#213 card
# probe): the 5090 (sm_120) does 232.0 bf16 TFLOPS and 566.9 fp8 TFLOPS, the
# 3080s (sm_86) have no fp8 tensor path at all. Scoring both cards in bf16
# gives 3.79 where the fp8 checkpoint runs at 8.6+, and the optimizer then
# proposes a 6,2,2-class split against a measured optimum of 10,1,1 (#296).
#
# Each lane is a MEASUREMENT or nothing. A card with no native fp8 unit still
# runs the checkpoint -- through Marlin, or through the dequant fallback --
# and those lanes are probed rather than estimated or left null. There is no
# datasheet number and no substitute constant anywhere on this path.
# ---------------------------------------------------------------------------

#: Dense bf16 matmul. The lane of an unquantized checkpoint, and the named
#: fallback for any format that has no lane table yet.
LANE_BF16 = "bf16"
#: Native fp8 e4m3 tensor cores via ``torch._scaled_mm`` (sm89+, Hopper,
#: Blackwell, MI300+).
LANE_FP8_NATIVE = "fp8_native"
#: Weight-only fp8 through ``gptq_marlin_gemm``. This is what sglang's dense
#: fp8 linear takes on sm80..88 -- see
#: ``fp8_utils.can_auto_enable_marlin_fp8`` -- so on an Ampere card serving an
#: fp8 checkpoint it is THE lane, not a curiosity.
LANE_FP8_MARLIN = "fp8_marlin"
#: Weight-only fp8 through dequantise-to-compute-dtype + dense matmul. The
#: route sglang takes when no fp8 GEMM of any kind is reachable, and the route
#: SGLANG_DETERMINISTIC_FP8_GEMM forces on sm80..88 (fp8 Marlin is run-to-run
#: nondeterministic there, #190). Weights stay fp8 in VRAM and are expanded
#: per forward, so at a prefill shape the expansion amortizes over the whole
#: token block -- which is exactly the shape this probe measures.
LANE_FP8_W8A16 = "fp8_w8a16"
#: Native NVFP4: E2M1 weights AND E2M1 activations on the block-scaled FP4
#: tensor path -- the fork's own sm_120a CUTLASS GEMM, or flashinfer's
#: cutlass/cutedsl runners on Blackwell datacentre parts. Reachable only where
#: ``fp4_utils.initialize_fp4_gemm_config`` resolves a non-Marlin backend, i.e.
#: sm_100, or sm_120 with the fork's JIT kernel genuinely importable (#323c).
LANE_NVFP4_NATIVE = "nvfp4_native"
#: Weight-only NVFP4 through ``gptq_marlin_gemm``: E2M1 is unpacked in-kernel
#: and multiplied on bf16 tensor cores. ``initialize_fp4_gemm_config`` resolves
#: it on sm_80..sm_89, and ``ModelOptNvFp4A16LinearMethod`` plus the
#: compressed-tensors W4A16-NVFP4 scheme take it on EVERY architecture,
#: Blackwell included -- a W4A16 family never reaches the native lane no matter
#: which card it lands on. That gap (measured band 2.6x on the reference rig's
#: 5090) is what a single per-rank score cannot represent.
LANE_NVFP4_MARLIN = "nvfp4_marlin"
#: Native INT8 W8A8 through ``sgl_kernel.int8_scaled_mm``: int8 weights AND
#: dynamically per-token-quantized int8 activations on the IMMA tensor path.
#: The kernel's SM dispatch is closed-ended (sm75, sm80..89, sm90, sm120+ since
#: #327a) and sm100/sm103 have no classic IMMA path at all, so whether a card
#: has this lane is a fact to be discovered by calling it -- which is precisely
#: what the probe does. Measured on the reference rig (#327): 678.00 TFLOPS on
#: the 5090 (1.21x its own fp8_native lane) and 177.74 / 182.23 on the 3080s
#: (2.96x / 3.05x their best fp8 lane, which is fp8_marlin -- sm_86 has no fp8
#: tensor path).
LANE_INT8_NATIVE = "int8_native"

#: Checkpoint compute format -> the lanes a card may take for it, in the order
#: the serving path tries them (fp8: native, then Marlin, then dequant --
#: mirroring ``fp8_utils.fp8_needs_dequant_fallback``). The dispatch takes the
#: FIRST lane that measured on that card, so a mixed rig scores each card on
#: the lane that card will run.
#:
#: This table is the extension point. AWQ / GPTQ-Marlin and GGUF are NOT
#: covered here: their lanes are different kernels (``gptq_marlin_gemm`` at 4
#: bits with a group-scale grid, MMQ/MMVQ for ggml types) and each needs its
#: own probe and its own evidence. A format with no entry falls back to the
#: bf16 lane WITH A WARNING -- never silently, because a bf16 number wearing a
#: quantized label is the defect this table exists to fix.
#:
#: The two ``nvfp4_*`` keys carry no probe yet -- their entries here are the
#: DISPATCH ORDER only, so that a mixed-precision checkpoint resolves the right
#: lane per family the moment the probes land. Until then a card takes the
#: named bf16 fallback below, and the fallback says the lane is unprobed rather
#: than telling the reader to re-run a probe that does not exist.
#: ``int8`` carries ONE lane on purpose. The fp8 family has a three-lane ladder
#: because a card without fp8 tensor cores still serves an fp8 checkpoint
#: through Marlin or a dequant fallback; INT8 W8A8 has no such fallback in this
#: tree (there is no ``dequant_int8_channel_weight``), so a card whose
#: ``int8_scaled_mm`` dispatch has no arm cannot serve the checkpoint at all.
#: Registering a second lane that the serving path could not take either would
#: make the plan lie; the existing "no lane measured on this card" branch of
#: ``rank_gemm_scores`` already produces the correct loud bf16 fallback.
_FORMAT_LANES: Dict[str, Tuple[str, ...]] = {
    "bf16": (LANE_BF16,),
    "fp8": (LANE_FP8_NATIVE, LANE_FP8_MARLIN, LANE_FP8_W8A16),
    "nvfp4_a4": (LANE_NVFP4_NATIVE, LANE_NVFP4_MARLIN),
    "nvfp4_a16": (LANE_NVFP4_MARLIN,),
    "int8": (LANE_INT8_NATIVE,),
}

#: Formats this table RECOGNISES but has no measured lane for. The distinction
#: is #606's: "we know this format and no lane of it is measurable here" is a
#: different statement from "we do not recognise this format at all", and a
#: reader of the bf16 fallback could not previously tell them apart.
#:
#: ``int8_a16`` (weight-only INT8) is here rather than in ``_FORMAT_LANES``
#: for the reason the comment above gives: the serving path has no weight-only
#: int8 arm, so registering a lane would make the plan lie.
FORMATS_WITHOUT_LANES: Dict[str, str] = {
    "int8_a16": "weight-only INT8 (W8A16): no weight-only int8 arm exists in "
    "this tree, so no lane can be measured for it",
}

#: Human-readable lane names for the plan log.
_LANE_LABELS = {
    LANE_BF16: "dense bf16",
    LANE_FP8_NATIVE: "fp8 native (_scaled_mm)",
    LANE_FP8_MARLIN: "fp8 Marlin (weight-only)",
    LANE_FP8_W8A16: "fp8 W8A16 dequant",
    LANE_NVFP4_NATIVE: "nvfp4 native (block-scaled FP4)",
    LANE_NVFP4_MARLIN: "nvfp4 Marlin (weight-only W4A16)",
    LANE_INT8_NATIVE: "int8 W8A8 native (int8_scaled_mm)",
}


def _time_gemm_tflops(dev, fn) -> float:
    """Timed TFLOPS for one GEMM lane at the probe shape.

    ``fn`` performs a single ``(M, K) x (K, N)`` product; the FLOP count is the
    same for every lane, so the returned numbers are directly comparable."""
    import torch

    for _ in range(_PROBE_GEMM_WARMUP):
        fn()
    torch.cuda.synchronize(dev)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(torch.cuda.current_stream(dev))
    for _ in range(_PROBE_GEMM_ITERS):
        fn()
    e.record(torch.cuda.current_stream(dev))
    torch.cuda.synchronize(dev)
    ms = s.elapsed_time(e) / _PROBE_GEMM_ITERS
    flops = 2.0 * _PROBE_GEMM_M * _PROBE_GEMM_K * _PROBE_GEMM_N
    return flops / (ms / 1e3) / 1e12


def _bench_gemm_tflops(dev) -> float:
    """Dense bf16 GEMM throughput (TFLOPS) at the probe shape."""
    import torch

    a = torch.randn(_PROBE_GEMM_M, _PROBE_GEMM_K, dtype=torch.bfloat16, device=dev)
    b = torch.randn(_PROBE_GEMM_K, _PROBE_GEMM_N, dtype=torch.bfloat16, device=dev)
    try:
        return _time_gemm_tflops(dev, lambda a=a, b=b: a @ b)
    finally:
        del a, b
        torch.cuda.empty_cache()


def _fp8_block_scale_shape(rows: int, cols: int) -> Tuple[int, int]:
    """Block-scale grid for a ``(rows, cols)`` fp8 weight, ``[block_n, block_k]``
    applied to the two dimensions in that order."""
    bn, bk = _PROBE_FP8_BLOCK
    return (rows + bn - 1) // bn, (cols + bk - 1) // bk


def _bench_gemm_fp8_native_tflops(dev) -> Tuple[Optional[float], str]:
    """Native fp8 e4m3 GEMM via ``torch._scaled_mm``, or ``(None, why)``.

    Per-tensor scales with ``use_fast_accum=True``: the configuration the fp8
    serving path uses, so the number prices the kernel that would actually run
    rather than a slower reference variant. Asked FUNCTIONALLY -- the lane is
    attempted, not inferred from a capability integer -- because the integer is
    ambiguous across vendors (gfx900 reports (9, 0), the same value as Hopper;
    see ``fp8_utils.fp8_native_gemm_available``)."""
    import torch

    if not hasattr(torch, "_scaled_mm"):
        return None, "this torch build has no torch._scaled_mm"
    a = b = scale = None
    try:
        a = torch.randn(
            _PROBE_GEMM_M, _PROBE_GEMM_K, dtype=torch.bfloat16, device=dev
        ).to(torch.float8_e4m3fn)
        # Column-major right operand: _scaled_mm requires mat2 to be
        # transposed-contiguous, and building it any other way fails at the
        # first call rather than measuring something.
        b = (
            torch.randn(_PROBE_GEMM_N, _PROBE_GEMM_K, dtype=torch.bfloat16, device=dev)
            .to(torch.float8_e4m3fn)
            .t()
        )
        scale = torch.ones((), dtype=torch.float32, device=dev)

        def fn(a=a, b=b, scale=scale):
            return torch._scaled_mm(
                a, b, scale, scale, out_dtype=torch.bfloat16, use_fast_accum=True
            )

        return _time_gemm_tflops(dev, fn), ""
    except Exception as ex:
        return None, f"native fp8 GEMM did not run: {type(ex).__name__}: {ex}"
    finally:
        del a, b, scale
        torch.cuda.empty_cache()


def _bench_gemm_fp8_marlin_tflops(dev) -> Tuple[Optional[float], str]:
    """Weight-only fp8 through the Marlin GEMM, or ``(None, why)``.

    Built with the SAME helpers the serving path uses
    (``marlin_utils_fp8.prepare_fp8_layer_for_marlin`` to repack the weight and
    permute the block scales, ``apply_fp8_marlin_linear`` to run it), so the
    measurement prices that kernel and not a re-implementation of it. On
    sm80..88 this is the lane a dense fp8 linear takes by default, which is
    what makes it the honest compute score for an Ampere rank serving an fp8
    checkpoint."""
    import torch

    layer = None
    x = None
    try:
        from sglang.srt.layers.quantization.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
            prepare_fp8_layer_for_marlin,
        )
    except Exception as ex:
        return None, f"fp8 Marlin kernels unavailable: {type(ex).__name__}: {ex}"
    try:
        k, n = _PROBE_GEMM_K, _PROBE_GEMM_N
        layer = torch.nn.Module()
        layer.orig_dtype = torch.bfloat16
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.weight_block_size = list(_PROBE_FP8_BLOCK)
        # size_k_first layout: the checkpoint's (k, n) weight, block scales on
        # the (k // block_k, ceil(n / block_n)) grid the preparer expects.
        layer.weight = torch.nn.Parameter(
            torch.randn(k, n, dtype=torch.bfloat16, device=dev).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        n_blocks, k_blocks = _fp8_block_scale_shape(n, k)
        layer.weight_scale = torch.nn.Parameter(
            torch.ones((k_blocks, n_blocks), dtype=torch.float32, device=dev),
            requires_grad=False,
        )
        prepare_fp8_layer_for_marlin(layer, size_k_first=True)
        x = torch.randn(_PROBE_GEMM_M, k, dtype=torch.bfloat16, device=dev)

        def fn(layer=layer, x=x, n=n, k=k):
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=n,
                size_k=k,
                bias=None,
            )

        return _time_gemm_tflops(dev, fn), ""
    except Exception as ex:
        return None, f"fp8 Marlin GEMM did not run: {type(ex).__name__}: {ex}"
    finally:
        del layer, x
        torch.cuda.empty_cache()


def _bench_gemm_w8a16_tflops(dev) -> Tuple[Optional[float], str]:
    """Weight-only fp8 through the DEQUANT lane, or ``(None, why)``.

    The weight stays fp8 in VRAM and is expanded to the compute dtype per
    forward, then a dense matmul runs on the card's bf16 units -- measured with
    the serving helper (``fp8_utils.dequant_fp8_block_weight``) rather than a
    stand-in, so the block-scale traffic and the temporary are both in the
    timing. The expansion is per FORWARD, not per token, so a prefill-shaped
    probe amortizes it over the whole token block; that is the shape the
    prefill objective needs, and it is why this lane is only mildly behind
    Marlin at prefill (measured -8.1 % on the reference rig) while it is far
    behind at batch-1 decode (-69.9 %, see ``fp8_utils._DequantCache``)."""
    import torch

    w = scale = x = None
    try:
        from sglang.srt.layers.quantization.fp8_utils import dequant_fp8_block_weight
    except Exception as ex:
        return None, f"fp8 dequant helper unavailable: {type(ex).__name__}: {ex}"
    try:
        k, n = _PROBE_GEMM_K, _PROBE_GEMM_N
        # Serving orientation: weight (n, k), activations (m, k) -> F.linear.
        w = torch.randn(n, k, dtype=torch.bfloat16, device=dev).to(
            torch.float8_e4m3fn
        )
        n_blocks, k_blocks = _fp8_block_scale_shape(n, k)
        scale = torch.ones((n_blocks, k_blocks), dtype=torch.float32, device=dev)
        x = torch.randn(_PROBE_GEMM_M, k, dtype=torch.bfloat16, device=dev)
        block = list(_PROBE_FP8_BLOCK)

        def fn(w=w, scale=scale, x=x, block=block):
            wd = dequant_fp8_block_weight(w, scale, block, torch.bfloat16)
            return torch.nn.functional.linear(x, wd)

        return _time_gemm_tflops(dev, fn), ""
    except Exception as ex:
        return None, f"fp8 W8A16 dequant GEMM did not run: {type(ex).__name__}: {ex}"
    finally:
        del w, scale, x
        torch.cuda.empty_cache()


def _bench_gemm_int8_native_tflops(dev) -> Tuple[Optional[float], str]:
    """Native INT8 W8A8 GEMM via ``sgl_kernel.int8_scaled_mm``, or ``(None, why)``.

    Per-token activation scales and per-channel weight scales -- the shape the
    W8A8 serving path produces (``per_token_quant_int8`` then
    ``int8_scaled_mm``), so the number prices the kernel that would actually
    run. The lane is asked FUNCTIONALLY: the kernel's SM dispatch is
    closed-ended and whether this card has a branch is a fact to be discovered
    by calling it, not inferred from a capability integer -- the scheme's own
    ``get_min_capability() == 80`` admits sm_100/sm_103, which have no classic
    IMMA path at all and would abort inside CUTLASS at the first forward.

    Landed unchanged from the standalone #320 script
    (``p320_int8_lane_probe.py``), which has run twice on the reference rig."""
    import torch

    try:
        from sgl_kernel import int8_scaled_mm
    except Exception as ex:
        return (
            None,
            f"sgl_kernel.int8_scaled_mm not importable: {type(ex).__name__}: {ex}",
        )

    a = b = sa = sb = None
    try:
        a = torch.randint(
            -127, 127, (_PROBE_GEMM_M, _PROBE_GEMM_K), dtype=torch.int8, device=dev
        )
        # Column-major right operand: the kernel asserts mat_b.stride(0) == 1,
        # so it is built as (N, K) and transposed, exactly like the fp8 native
        # lane builds its own mat2.
        b = torch.randint(
            -127, 127, (_PROBE_GEMM_N, _PROBE_GEMM_K), dtype=torch.int8, device=dev
        ).t()
        sa = torch.ones(_PROBE_GEMM_M, dtype=torch.float32, device=dev)
        sb = torch.ones(_PROBE_GEMM_N, dtype=torch.float32, device=dev)

        def fn(a=a, b=b, sa=sa, sb=sb):
            return int8_scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16, bias=None)

        # One call outside the timing harness: a dispatch that has no branch
        # for this card must surface as a note here, not as a crash inside the
        # warmup loop.
        fn()
        return _time_gemm_tflops(dev, fn), ""
    except Exception as ex:
        return None, f"int8 GEMM did not run: {type(ex).__name__}: {ex}"
    finally:
        del a, b, sa, sb
        torch.cuda.empty_cache()


class ProbeEnvironmentError(RuntimeError):
    """The probe's OWN interpreter is missing something the ``lanes`` group
    needs to measure honestly -- as opposed to a genuine per-card GEMM
    failure (task #310).

    ``LANE_FP8_MARLIN`` (``_bench_gemm_fp8_marlin_tflops``) runs through
    ``marlin_utils_fp8``, which resolves its scalar types via
    ``sglang.srt.layers.quantization.utils.get_scalar_types()``. That helper
    falls back to a ``MockScalarTypes`` stand-in when ``sgl_kernel`` is not
    importable -- needed elsewhere so an sgl_kernel-less import does not
    explode -- and the Marlin lane then calls real sgl_kernel APIs
    (``b_q_type.id``) against that mock, gets an ``AttributeError``, and the
    lane probe's ``except Exception`` catches it and returns it as the lane's
    REASON: ``"fp8 Marlin GEMM did not run: AttributeError: ..."``. That
    string is indistinguishable, downstream, from a genuine architecture
    limitation -- so a venv without sgl_kernel silently wrote "this card
    cannot do fp8 Marlin" into the cached profile for EVERY card, including
    ones (e.g. an sm89+ card) that run the lane natively. Once cached, the
    wrong verdict survives until someone notices the number is off and
    manually re-probes.

    So this is a hard, up-front gate: an interpreter without real sgl_kernel
    scalar types must not run the ``lanes`` group AT ALL, and must not write
    a profile -- raised before ``run_probe`` touches a single card."""


def _check_lane_probe_environment() -> None:
    """Refuse to probe the ``lanes`` group unless THIS interpreter has real
    sgl_kernel scalar types. See ``ProbeEnvironmentError`` for why a missing
    or mocked sgl_kernel must abort the whole probe rather than be recorded
    as a card fact."""
    interpreter = sys.executable
    try:
        import sgl_kernel  # noqa: F401
    except ImportError as ex:
        raise ProbeEnvironmentError(
            "the 'lanes' probe group needs sgl_kernel, which is not "
            f"importable in {interpreter} ({type(ex).__name__}: {ex}). Run "
            "the probe with the interpreter sglang actually serves from (its "
            "venv ships sgl_kernel), or pass --groups with 'lanes' left out "
            "to measure only the groups this interpreter can measure "
            "honestly."
        ) from ex

    from sglang.srt.layers.quantization.utils import get_scalar_types

    _, probed_scalar_types = get_scalar_types()
    if type(probed_scalar_types).__name__ == "MockScalarTypes":
        raise ProbeEnvironmentError(
            f"sgl_kernel is importable in {interpreter}, but "
            "sglang.srt.layers.quantization.utils.get_scalar_types() still "
            "returned its MockScalarTypes fallback -- sgl_kernel.scalar_type "
            "did not load cleanly in this interpreter. Same risk as a "
            "missing sgl_kernel (a lane failure here would be an "
            "environment artifact mis-recorded as a card fact), so the "
            "'lanes' probe group refuses to run here."
        )


#: lane -> probe. Every entry returns ``(tflops or None, note)``; a lane that
#: cannot run on a card stores its REASON, never a substitute number.
_LANE_PROBES = {
    LANE_FP8_NATIVE: _bench_gemm_fp8_native_tflops,
    LANE_FP8_MARLIN: _bench_gemm_fp8_marlin_tflops,
    LANE_FP8_W8A16: _bench_gemm_w8a16_tflops,
    LANE_INT8_NATIVE: _bench_gemm_int8_native_tflops,
}


def _bench_gemm_lanes(dev) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Every quantized GEMM lane this card can run, at the probe shape.

    Returns ``({lane: tflops}, {lane: why_absent})``. The dense bf16 lane is
    not in here -- it is ``gemm_tflops``, probed unconditionally. Lanes are
    measured for every card regardless of which checkpoint is being planned:
    the profile is cached per RIG, and re-probing it per checkpoint would cost
    a probe on every boot that changes the model."""
    values: Dict[str, float] = {}
    notes: Dict[str, str] = {}
    for lane, probe in _LANE_PROBES.items():
        tflops, note = probe(dev)
        if tflops is not None:
            values[lane] = round(tflops, 2)
        else:
            notes[lane] = note
    return values, notes


def _time_best_gbs(dev, fn, moved_bytes: float, iters: int = 40) -> float:
    """Effective GB/s for a bandwidth-bound kernel: the BEST (min-time) of a few
    iters (best-of rejects scheduling/clock-warmup noise better than a mean for
    a memory microbenchmark). ``moved_bytes`` is the DRAM traffic per call."""
    import torch

    for _ in range(8):
        fn()
    torch.cuda.synchronize(dev)
    best_ms = float("inf")
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(torch.cuda.current_stream(dev))
        fn()
        e.record(torch.cuda.current_stream(dev))
        torch.cuda.synchronize(dev)
        best_ms = min(best_ms, s.elapsed_time(e))
    return moved_bytes / 1e9 / (best_ms / 1e3)


@dataclasses.dataclass
class MembwRates:
    """The three rates the memory-bandwidth probe measures, kept APART.

    ``read_gbs`` / ``copy_gbs`` are streaming kernels: they establish what the
    card's DRAM path can do when nothing else limits it. ``gemv_gbs`` is the
    decode-shaped weight read -- one matrix pulled through once against a
    single activation vector, which is what a bs=1 decode step does to every
    weight tensor it touches. The three answer different questions, so they
    are reported separately; collapsing them to a maximum returns the largest,
    which is the streaming peak on every card measured here, and the decode
    number is then lost precisely where it was supposed to be used."""

    read_gbs: float
    copy_gbs: float
    gemv_gbs: float

    @property
    def streaming_peak_gbs(self) -> float:
        """Best rate a pure stream reaches -- the card's bandwidth score."""
        return max(self.read_gbs, self.copy_gbs)


def _bench_membw_rates(dev) -> MembwRates:
    """MEASURED device memory bandwidth (GB/s), per kernel — a genuine
    bandwidth-bound probe, NOT a nameplate spec. Three kernels over a working
    set well past L2: a read-only reduction, a copy, and the decode-shaped
    GEMV weight read.

    Measured rates land BELOW the nameplate peak (~1.66 TB/s streaming on a
    5090 vs ~1.79 nameplate, ~0.72 TB/s on a 3080 vs ~0.76) — expected, and
    exactly why the roofline prefers this probe over the reference table for
    cards on-box.

    Shape stability (measured on the reference rig, sweeping the GEMV row
    count over 98304..163840 at fixed K): each rate repeats to within 0.3 %,
    and the 5090:3080 GEMV ratio stays in 2.131..2.145. The size matters: at
    per-layer granularity (~32 MiB matrices) the same measurement swings
    1.98..2.58 depending on how the row count happens to divide, and a 64 MiB
    matrix is L2-resident on a 5090 and reads at 2.2 TB/s, past its own DRAM
    peak. One large matrix is the shape that measures DRAM rather than kernel
    tiling luck, which is why the probe uses it."""
    import torch

    n = _PROBE_GEMV_ROWS * _PROBE_GEMV_K  # ~0.67 G elems -> ~1.34 GB bf16
    a = torch.randn(n, dtype=torch.bfloat16, device=dev)
    b = torch.empty(n, dtype=torch.bfloat16, device=dev)
    x = torch.randn(1, _PROBE_GEMV_K, dtype=torch.bfloat16, device=dev)
    w = a.view(_PROBE_GEMV_ROWS, _PROBE_GEMV_K)
    nbytes = n * 2
    # Tensors bound as defaults, not captured: the buffers are freed below,
    # and a late-binding closure over a deleted name is only ever a trap.
    rates = MembwRates(
        read_gbs=_time_best_gbs(dev, lambda a=a: a.sum(), nbytes),
        copy_gbs=_time_best_gbs(dev, lambda a=a, b=b: b.copy_(a), nbytes * 2),
        gemv_gbs=_time_best_gbs(
            dev,
            lambda x=x, w=w: torch.nn.functional.linear(x, w),
            nbytes,
        ),
    )
    del a, b, x, w
    torch.cuda.empty_cache()
    return rates


def _bench_membw_gbs(dev) -> float:
    """The card's STREAMING bandwidth score (GB/s) — ``_bench_membw_rates``
    reduced to the one number the bandwidth-score consumers want (the vocab /
    KV-speed weighting, the power calibration's decode-relevant active load).
    The decode roofline does NOT use this; it uses the GEMV rate, see
    ``PerfCostModel.decode_bw_basis``."""
    return _bench_membw_rates(dev).streaming_peak_gbs


#: Environment the link workers must NOT inherit. Every one of these steers
#: torch's ``env://`` rendezvous, and the probe is a private, single-node,
#: three-second process group that has nothing to do with whatever distributed
#: job the launching shell was configured for. Inheriting any of them points
#: the probe's rendezvous somewhere it will never complete -- see
#: ``_link_rendezvous_env``.
_LINK_ENV_TO_CLEAR = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
)


def _free_tcp_port() -> int:
    """A currently free localhost TCP port, for the probe's own rendezvous."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _link_rendezvous_env(port: int) -> None:
    """Pin the link probe's rendezvous to a private localhost endpoint.

    ROOT CAUSE, task #303. This used to be ``os.environ.setdefault``, on a
    hardcoded port. Both halves are defects:

      * ``setdefault`` means an inherited ``MASTER_ADDR`` wins. Rank 0 then
        binds the store on the WILDCARD address (torch's TCPStore server
        ignores the hostname when listening) and waits for its workers, while
        the workers dial the inherited address and never reach it. Every rank
        sits in the ``TCPStore`` constructor for the full process-group
        timeout -- 600 s with the NCCL default -- and the probe subprocess is
        then killed by its own outer timeout. Reproduced on CPU with
        ``MASTER_ADDR=192.0.2.7``: all ranks block at
        ``rendezvous.py:_create_c10d_store``, byte-for-byte the py-spy
        signature captured on the rig.
      * a hardcoded port collides with any concurrent or orphaned probe, and
        the collision surfaces as a bind error on rank 0 that kills the whole
        probe.
      * ``TORCHELASTIC_USE_AGENT_STORE=True`` makes EVERY rank a store client,
        so nobody hosts the store and all of them wait out the timeout.

    So the endpoint is dictated, not defaulted: the parent picks a free port
    and every steering variable is removed from the child environment first."""
    for var in _LINK_ENV_TO_CLEAR:
        os.environ.pop(var, None)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)


def _link_worker(rank: int, world: int, results, port: int, timeout_s: float) -> None:
    """NCCL pairwise link probe (adapted from m20_nccl_bench): P2P 1 MiB
    bandwidth per GPU pair + small all-reduce latency over the full group."""
    import torch
    import torch.distributed as dist

    _link_rendezvous_env(port)
    # Rank-local condition BEFORE the collective: prove this rank's own card
    # answers, and publish that it got that far. A rank that cannot allocate
    # on its device must fail here with its own error, not disappear into a
    # group wait that reports nothing about which rank is broken; and when the
    # group DOES hang, the parent can name the ranks that reached the
    # rendezvous versus the ones that never arrived.
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    torch.zeros(1, device=dev)
    results[f"reached_rendezvous:{rank}"] = True

    # Explicit, finite timeout: torch's default for NCCL is 600 s, which is
    # the entire cost this probe is supposed to avoid. The group is local and
    # tiny -- if it does not form within the phase budget it is not going to.
    from datetime import timedelta

    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=max(5.0, timeout_s)),
    )
    out = {}

    def bench(fn, iters=100, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1000.0  # us

    t = torch.randn(5120, dtype=torch.bfloat16, device=dev)
    out["ar_10kb_us"] = bench(lambda: dist.all_reduce(t))
    big = torch.randn(512 * 1024, dtype=torch.bfloat16, device=dev)
    out["ar_1mb_us"] = bench(lambda: dist.all_reduce(big))

    numel = 512 * 1024  # 1 MiB bf16
    for a in range(world):
        for b in range(a + 1, world):
            buf = torch.randn(numel, dtype=torch.bfloat16, device=dev)
            if rank == a:
                fn = lambda buf=buf, b=b: dist.send(buf, b)
            elif rank == b:
                fn = lambda buf=buf, a=a: dist.recv(buf, a)
            else:
                fn = lambda: None
            us = bench(fn, iters=60)
            if rank == a:
                out[f"p2p_{a}_{b}_gbs"] = numel * 2 / 1e9 / (us / 1e6)
    results[rank] = out
    dist.barrier()
    dist.destroy_process_group()


def _link_timeout_s() -> float:
    from sglang.srt.environ import envs

    return float(envs.SGLANG_PERF_PROBE_LINK_TIMEOUT_S.get())


def probe_link_matrix(
    gpus: Sequence[dict], timeout_s: Optional[float] = None
) -> Tuple[Dict[str, dict], str]:
    """Pairwise NCCL link matrix, under a HARD wall-clock cap.

    Returns ``(links, reason)``; ``reason`` is empty when the phase completed
    and names the failure otherwise, for the caller to store next to the empty
    table (same mechanic as ``gemm_lane_notes``).

    The cap exists because this is the only phase of the probe that waits on
    something other than this rig's own hardware: it forms a process group, so
    a rendezvous that cannot complete blocks for the process-group timeout
    rather than failing. That is a boot-time cost paid on the critical path,
    and no measurement is worth an unbounded wait -- the per-card numbers are
    already in hand at this point, and the link matrix only refines the
    communication term of the plan."""
    import torch
    import torch.multiprocessing as mp

    links: Dict[str, dict] = {}
    world = int(torch.cuda.device_count())
    if world <= 1:
        return links, ""
    budget = _link_timeout_s() if timeout_s is None else float(timeout_s)
    if budget <= 0:
        return links, "link matrix disabled (link timeout <= 0)"

    port = _free_tcp_port()
    mgr = mp.Manager()
    results = mgr.dict()
    ctx = mp.spawn(
        _link_worker,
        args=(world, results, port, budget),
        nprocs=world,
        join=False,
    )
    deadline = time.time() + budget
    reason = ""
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                reached = sorted(
                    int(k.split(":")[1])
                    for k in list(results.keys())
                    if isinstance(k, str) and k.startswith("reached_rendezvous:")
                )
                missing = [r for r in range(world) if r not in reached]
                reason = (
                    f"link matrix timed out after {budget:.0f} s "
                    f"(rendezvous 127.0.0.1:{port}, world {world}; ranks that "
                    f"reached the rendezvous: {reached or 'none'}"
                    + (f", ranks that never arrived: {missing}" if missing else "")
                    + "). Per-card measurements are kept; the plan falls back "
                    "to its uniform link assumption. Raise "
                    "SGLANG_PERF_PROBE_LINK_TIMEOUT_S to allow more, or set "
                    "SGLANG_PERF_PROBE_SKIP_LINKS=1 to stop attempting it."
                )
                break
            # join() returns True once every worker exited cleanly, raises if
            # one died, and returns False on timeout -- so this loop is bounded
            # by the deadline in every branch.
            if ctx.join(timeout=min(1.0, remaining)):
                break
    except Exception as ex:
        reason = f"link matrix failed: {type(ex).__name__}: {ex}"
    finally:
        _terminate_spawn_context(ctx)
        # Snapshot before the manager goes away, then shut it down: the probe
        # process exits right after this and must not leave a manager behind.
        harvest = {k: dict(v) for k, v in results.items() if isinstance(v, dict)}
        try:
            mgr.shutdown()
        except Exception:
            pass

    if reason:
        return {}, reason

    by_idx = {g["cuda_index"]: g["uuid"] for g in gpus}
    r0 = harvest.get(0, {})
    for a in range(world):
        ra = harvest.get(a, {})
        for b in range(a + 1, world):
            if a not in by_idx or b not in by_idx:
                continue
            key = "|".join(sorted([by_idx[a], by_idx[b]]))
            gbs = ra.get(f"p2p_{a}_{b}_gbs")
            if gbs is not None:
                links[key] = {"p2p_gbs": round(gbs, 2)}
    if "ar_10kb_us" in r0:
        links["__group__"] = {
            "ar_10kb_us": round(r0["ar_10kb_us"], 1),
            "ar_1mb_us": round(r0["ar_1mb_us"], 1),
        }
    return links, ""


def _terminate_spawn_context(ctx, grace_s: float = 5.0) -> None:
    """Tear down an ``mp.spawn`` context: SIGTERM, a bounded grace period, then
    SIGKILL. Bounded at every step -- a probe that hung must not leave workers
    behind holding CUDA contexts on the cards the server is about to load."""
    procs = list(getattr(ctx, "processes", []) or [])
    for p in procs:
        if p.is_alive():
            p.terminate()
    deadline = time.time() + grace_s
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.time()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join(timeout=1.0)


def _probe_one_gpu(g: dict, groups: Sequence[str]) -> dict:
    """Run the requested probe groups on one card and return the fields they
    measured. Groups run in the fixed order of the full probe."""
    import torch

    dev = torch.device(f"cuda:{g['cuda_index']}")
    torch.cuda.set_device(dev)
    fields: Dict[str, object] = {}
    if PROBE_GROUP_GEMM in groups:
        fields["gemm_tflops"] = round(_bench_gemm_tflops(dev), 2)
    if PROBE_GROUP_LANES in groups:
        lane_tflops, lane_notes = _bench_gemm_lanes(dev)
        # Quantized GEMM lanes, and the reason for each one that is absent.
        # The prefill objective picks per card by the checkpoint's format (see
        # rank_gemm_scores); a missing lane must read as a missing
        # MEASUREMENT, which is why the note is stored next to it.
        fields["gemm_lanes"] = lane_tflops
        fields["gemm_lane_notes"] = lane_notes
    if PROBE_GROUP_MEMBW in groups:
        rates = _bench_membw_rates(dev)
        # Unchanged meaning and unchanged consumers: the card's streaming
        # bandwidth score.
        fields["membw_gbs"] = round(rates.streaming_peak_gbs, 1)
        # The three rates kept apart. membw_gemv_gbs is the decode roofline's
        # divisor; the other two are what it is judged against (see
        # _PROBE_GEMV_SATURATION_FLOOR / _PROBE_GEMV_CACHE_CEILING).
        fields["membw_read_gbs"] = round(rates.read_gbs, 1)
        fields["membw_copy_gbs"] = round(rates.copy_gbs, 1)
        fields["membw_gemv_gbs"] = round(rates.gemv_gbs, 1)
    return fields


def _write_profile(profile: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile, f, indent=1)
    os.replace(tmp, out_path)


def run_probe(
    out_path: str,
    groups: Optional[Sequence[str]] = None,
    base: Optional[dict] = None,
) -> dict:
    """Execute the stage-0 probe on every visible CUDA device and write the
    JSON profile to ``out_path``. Meant to run inside the dedicated probe
    subprocess (see ``get_hardware_profile``).

    ``groups`` selects which probe groups to measure; ``None`` is the full
    probe (all groups plus the link matrix). Passing a subset together with
    ``base`` is the LAZY TOP-UP path: only the named groups are measured, every
    other field -- including the link matrix, the slowest phase by far -- is
    carried over from ``base``. That is what makes an additive PROFILE_VERSION
    bump cost one GEMM pass instead of a full stage 0."""
    t0 = time.time()
    gpus, driver = _nvml_gpu_inventory()
    topup = groups is not None
    groups = list(_PROBE_GROUP_FIELDS) if groups is None else list(groups)
    base_gpus = dict((base or {}).get("gpus") or {})

    per_gpu: Dict[str, dict] = {}
    for g in gpus:
        entry = dict(base_gpus.get(g["uuid"]) or {})
        entry.update(
            {
                "name": g["name"],
                "cuda_index": g["cuda_index"],
                "total_mib": g["total_mib"],
            }
        )
        entry.update(_probe_one_gpu(g, groups))
        per_gpu[g["uuid"]] = entry

    from sglang.srt.environ import envs

    notes: Dict[str, str] = dict((base or {}).get(PROFILE_NOTES_KEY) or {})
    if topup:
        # The top-up never re-runs the network phase: its whole point is to
        # add per-card fields to a profile whose link matrix already measured.
        links = dict((base or {}).get("links") or {})
    elif bool(envs.SGLANG_PERF_PROBE_SKIP_LINKS.get()):
        links = {}
        notes["links"] = "link matrix skipped (SGLANG_PERF_PROBE_SKIP_LINKS=1)"
    else:
        links, link_reason = probe_link_matrix(gpus)
        if link_reason:
            notes["links"] = link_reason
            logger.warning("auto-performance probe: %s", link_reason)
        else:
            notes.pop("links", None)

    profile = dict(base or {})
    profile.update(
        {
            "version": PROFILE_VERSION,
            "driver": driver,
            "uuids": sorted(per_gpu),
            "gpus": per_gpu,
            "links": links,
            "probe_seconds": round(time.time() - t0, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if topup:
        profile["topup_groups"] = groups
    if notes:
        profile[PROFILE_NOTES_KEY] = notes
    else:
        profile.pop(PROFILE_NOTES_KEY, None)
    _write_profile(profile, out_path)
    return profile


#: Marker put in front of every line of a quoted SUBPROCESS log. The probe's
#: own failure is caught, named and recovered from here; the traceback is
#: quoted only as evidence. Without the marker the quoted traceback reads to a
#: log scanner exactly like a crash of the server itself -- which is how a
#: deliberately killed probe was scored as a boot failure (#303 part 4). The
#: battery's scanner knows this token; see
#: scripts/gpu_battery/checks/check_common.py.
QUOTED_SUBLOG_PREFIX = "[probe-subprocess] "

#: Marker the subprocess entry point prints in front of a ``ProbeEnvironmentError``
#: (task #310), so ``_run_probe_subprocess`` can tell "this interpreter is
#: missing something the probe needs" apart from an ordinary probe crash. The
#: distinction matters one level up: a lazy top-up that fails for this reason
#: must record its gap as ``NOTE_CLASS_ENVIRONMENT`` (logged, never
#: persisted), not ``NOTE_CLASS_CARD`` (persisted) -- see ``_migrated_profile``.
PROBE_ENV_ERROR_PREFIX = "PROBE_ENVIRONMENT_ERROR: "


def _quote_sublog(text: str, limit: int = 2000) -> str:
    """Prefix every line of a captured subprocess log with the marker above."""
    tail = (text or "")[-limit:]
    return "\n".join(QUOTED_SUBLOG_PREFIX + line for line in tail.splitlines())


def _probe_timeout_s() -> float:
    from sglang.srt.environ import envs

    return float(envs.SGLANG_PERF_PROBE_TIMEOUT_S.get())


def _run_probe_subprocess(path: str, groups: Optional[Sequence[str]]) -> Optional[str]:
    """Run the probe in its own process (so the launcher stays free of CUDA
    contexts). Returns None on success, or a one-line reason on failure.

    A failure caused by ``ProbeEnvironmentError`` (task #310: the interpreter
    is missing something the probe needs, not a card fact) is returned with
    ``PROBE_ENV_ERROR_PREFIX`` kept in front of it, so a caller that records
    the failure as a profile note (``_migrated_profile``) can classify it as
    ``NOTE_CLASS_ENVIRONMENT`` rather than ``NOTE_CLASS_CARD``."""
    cmd = [sys.executable, "-m", "sglang.srt.uneven_perf", "--probe", "--out", path]
    if groups:
        cmd += ["--groups", ",".join(groups)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_probe_timeout_s(),
            check=False,
        )
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    if proc.returncode != 0:
        stderr = proc.stderr or proc.stdout or ""
        logger.warning(
            "auto-performance: hardware probe failed (rc=%d); the quoted lines "
            "below are the PROBE's log, not this server's:\n%s",
            proc.returncode,
            _quote_sublog(stderr),
        )
        env_line = next(
            (
                line
                for line in stderr.splitlines()
                if line.startswith(PROBE_ENV_ERROR_PREFIX)
            ),
            None,
        )
        if env_line is not None:
            return env_line
        return f"probe subprocess exited with rc={proc.returncode}"
    return None


def _load_profile(path: str, driver: str, uuids: Sequence[str]) -> Optional[dict]:
    """A cached profile from ``path`` whose rig key matches, else None. The
    schema version is NOT checked here -- the caller decides between using and
    migrating it."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            profile = json.load(f)
    except Exception as e:
        logger.warning(
            "auto-performance: could not read cached profile %s (%s).", path, e
        )
        return None
    if profile.get("driver") != driver or profile.get("uuids") != sorted(uuids):
        return None
    return profile


def _migrated_profile(
    driver: str, uuids: Sequence[str], path: str
) -> Tuple[Optional[dict], str]:
    """Migrate the newest older-version cache file for this rig, topping up
    only the fields the newer versions added.

    Returns ``(profile or None, source description)``. The top-up runs the
    probe GROUPS those fields belong to and NOTHING else -- in particular not
    the pairwise link matrix, which the old profile already measured and which
    is the phase that can hang. If the top-up fails, the migrated profile is
    still returned: every value the old version measured is valid, and the
    consumers of the added fields already report a named gap rather than
    substituting a number."""
    for old_version, old_path in legacy_profile_paths(uuids, driver):
        old = _load_profile(old_path, driver, uuids)
        if old is None or int(old.get("version") or 0) != old_version:
            continue
        profile, missing = migrate_profile(old)
        gaps = sorted({f for fields in missing.values() for f in fields})
        if not gaps:
            _write_profile(profile, path)
            return profile, f"migrated v{old_version} cache ({old_path})"
        groups = probe_groups_for_fields(gaps)
        logger.info(
            "auto-performance: migrating the cached v%d hardware profile to v%d "
            "(%s) -- every measured value is kept; only the added field(s) %s "
            "are re-probed, which runs the %s probe group(s) and NOT the "
            "pairwise link matrix.",
            old_version,
            PROFILE_VERSION,
            old_path,
            ", ".join(gaps),
            ", ".join(groups),
        )
        # Stage the migrated profile at the new path first: the top-up
        # subprocess reads it as its base and merges into it, so a failed
        # top-up still leaves the carried-over values cached.
        _write_profile(profile, path)
        failure = _run_probe_subprocess(path, groups)
        if failure is None:
            topped = _load_profile(path, driver, uuids)
            if topped is not None:
                return (
                    topped,
                    f"migrated v{old_version} cache + {'/'.join(groups)} top-up "
                    f"({topped.get('probe_seconds', '?')} s)",
                )
            failure = "top-up wrote no readable profile"
        # A failure that carries PROBE_ENV_ERROR_PREFIX is a fact about the
        # interpreter that ran the top-up subprocess (task #310), not about
        # any card -- log it, but do not let it survive into the profile
        # file (see NOTE_CLASS_ENVIRONMENT / _profile_note).
        is_env_error = failure.startswith(PROBE_ENV_ERROR_PREFIX)
        note_class = NOTE_CLASS_ENVIRONMENT if is_env_error else NOTE_CLASS_CARD
        for field in gaps:
            _profile_note(
                profile,
                field,
                f"added in profile version {PROFILE_VERSION}; the lazy top-up "
                f"probe did not run ({failure}). Re-probe with "
                f"SGLANG_PERF_REPROBE=1 to measure it.",
                note_class=note_class,
            )
        logger.warning(
            "auto-performance: the lazy top-up probe failed (%s); keeping the "
            "migrated v%d profile without %s.",
            failure,
            old_version,
            ", ".join(gaps),
        )
        _write_profile(profile, path)
        return profile, f"migrated v{old_version} cache (top-up failed)"
    return None, ""


def get_hardware_profile() -> Tuple[Optional[dict], str, List[dict]]:
    """The rig's hardware profile: from the cache when the (sorted GPU UUIDs,
    driver version) key matches, otherwise migrated from an older schema
    version's cache, otherwise via a fresh probe subprocess (isolated so the
    launcher process stays free of CUDA contexts).

    Returns (profile or None, source description, per-CUDA-device inventory).
    SGLANG_PERF_REPROBE=1 forces a re-probe."""
    gpus, driver = _nvml_gpu_inventory()
    uuids = [g["uuid"] for g in gpus]
    path = profile_cache_path(uuids, driver)

    from sglang.srt.environ import envs

    force = bool(envs.SGLANG_PERF_REPROBE.get())
    reason = "forced by SGLANG_PERF_REPROBE=1" if force else "no cached profile"
    if not force:
        profile = _load_profile(path, driver, uuids)
        if profile is not None and profile.get("version") == PROFILE_VERSION:
            return profile, f"cache ({path})", gpus
        if profile is not None:
            reason = f"cached profile {path} has a stale key"
            logger.warning("auto-performance: %s; re-probing.", reason)
        else:
            migrated, source = _migrated_profile(driver, uuids, path)
            if migrated is not None:
                return migrated, source, gpus

    logger.info(
        "auto-performance: running the stage-0 hardware micro-probe (%s; "
        "GEMM per card in every reachable lane (bf16 + the quantized ones) "
        "+ memory bandwidth, pairwise NCCL link matrix under a %.0f s cap; "
        "a few seconds, cached to %s afterwards)...",
        reason,
        _link_timeout_s(),
        path,
    )
    failure = _run_probe_subprocess(path, None)
    if failure is None:
        profile = _load_profile(path, driver, uuids)
        if profile is not None:
            return (
                profile,
                f"fresh probe ({profile.get('probe_seconds', '?')} s)",
                gpus,
            )
        failure = "probe wrote no readable profile"
    logger.warning("auto-performance: hardware probe failed (%s).", failure)
    return None, "probe failed", gpus


def get_cached_hardware_profile() -> Tuple[Optional[dict], List[dict]]:
    """Cache-only variant of ``get_hardware_profile``: returns the cached
    profile when its (sorted GPU UUIDs, driver, version) key matches, else
    (None, inventory). NEVER triggers a probe -- used by consumers that only
    want opportunistic access to the measured scores (--rank-vocab-ratio
    auto), where a multi-second probe would be a surprising side effect.

    An older schema version's cache is MIGRATED in memory (values carried
    over, added fields simply absent) rather than ignored: the migration
    itself measures nothing, and dropping a rig's measured rates over a field
    this caller does not read would be a loss for no gain. Nothing is written
    back -- writing is the probing path's job."""
    gpus, driver = _nvml_gpu_inventory()
    uuids = [g["uuid"] for g in gpus]
    profile = _load_profile(profile_cache_path(uuids, driver), driver, uuids)
    if profile is not None and profile.get("version") == PROFILE_VERSION:
        return profile, gpus
    for old_version, old_path in legacy_profile_paths(uuids, driver):
        old = _load_profile(old_path, driver, uuids)
        if old is not None and int(old.get("version") or 0) == old_version:
            migrated, _ = migrate_profile(old)
            return migrated, gpus
    return None, gpus


# ---------------------------------------------------------------------------
# Format dispatch: checkpoint quantization -> per-card compute lane.
# ---------------------------------------------------------------------------


def _quant_config(cfg: dict) -> dict:
    """The checkpoint's ``quantization_config``, from the top level or from
    ``text_config`` (where VL checkpoints keep it). ``{}`` when unquantized."""
    qc = cfg.get("quantization_config")
    if not qc:
        text = cfg.get("text_config")
        if isinstance(text, dict):
            qc = text.get("quantization_config")
    return qc or {}


def _is_fp8_like(method: str, fmt: str) -> bool:
    """Whether a ``(quant_method, format)`` pair denotes an fp8 weight scheme.

    One predicate, two consumers: the byte model (``_config_quant_bpp``, which
    charges 1 B/weight for these) and the lane dispatch below. Splitting them
    would let the planner size a checkpoint as fp8 while scoring it as
    something else."""
    return method == "fp8" or "float8" in method or "float" in fmt or "fp8" in fmt


def _is_int8_w8a8_like(method: str, fmt: str, qc: dict) -> bool:
    """Whether a ``quantization_config`` denotes a genuine INT8 **W8A8** scheme
    -- the one ``sgl_kernel.int8_scaled_mm`` serves.

    False for weight-only INT8 (W8A16): compressed-tensors writes those as
    ``format: "pack-quantized"`` with no ``input_activations`` block at all,
    the weight is dequantized before a bf16 matmul, and none of the int8
    tensor-core advantage this lane measures applies to them.

    The ``input_activations`` block is trusted over the ``format`` string
    because it is what actually decides which kernel runs, and 8-bit INT
    activations are required so that a 4-bit or float variant cannot take this
    lane by accident. The standalone ``w8a8_int8`` method carries no
    ``config_groups`` -- its name in ``quant_method`` IS the declaration."""
    if method not in ("compressed-tensors", "compressed_tensors", "w8a8_int8"):
        return False
    if method == "w8a8_int8":
        return True
    if "pack-quantized" in fmt:
        return False
    for group in (qc.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        acts = (group or {}).get("input_activations") or {}
        try:
            wbits = int(weights.get("num_bits") or 0)
            abits = int(acts.get("num_bits") or 0)
        except (TypeError, ValueError):
            continue
        wtype = str(weights.get("type") or "").lower()
        atype = str(acts.get("type") or "").lower()
        if wbits == 8 and abits == 8 and wtype == "int" and atype in ("int", ""):
            return True
    return False


# ---------------------------------------------------------------------------
# Compute families: one score per rank is not enough.
#
# For fp8 -- one scheme applied to every Linear -- a single per-rank number is
# exactly right, and that is the only case the score derivation had to serve.
# A MIXED_PRECISION checkpoint breaks it on the SAME card: nvidia's Qwen3.6-27B
# NVFP4 (V1) keeps attention and the GDN in_proj at fp8 while the MLP is
# W4A16-NVFP4, so a 5090 runs attention on the native fp8 path and the MLP on
# Marlin -- a measured band of 2.6x INSIDE one rank, which a scalar cannot
# express at all (ANALYSE_321 sec. 8.3).
#
# The dimension added here is (rank, family). It is derived, per family, from
# the format that family's own modules declare in the checkpoint config, and
# resolved against the SAME per-card lane table the scalar path uses -- so the
# lane a family is scored on is the lane the serving path would resolve for
# that card x that family (``fp4_utils.initialize_fp4_gemm_config`` /
# ``get_fp4_gemm_runner_backend`` for the FP4 families, mirrored here by lane
# ORDER rather than by calling the resolver: the planner runs pre-boot in one
# process and must answer for every rank, while the resolver answers only for
# the device it is called on).
#
# No PROFILE_VERSION bump. The family dimension lives entirely in the score
# DERIVATION; the profile keeps its v3 ``gemm_lanes`` / ``gemm_lane_notes``
# fields and only gains new KEYS inside them as lanes are added. Bumping the
# version changes the cache key and forces every rig to re-probe the pairwise
# link matrix -- the 600 s/boot phase of the #303 incident -- for a change the
# cached measurements are already valid under.
# ---------------------------------------------------------------------------

#: Dense MLP (gate/up/down projections).
GEMM_FAMILY_MLP = "mlp"
#: Attention projections plus the GDN/linear-attention in_proj/out_proj. One
#: family because every checkpoint seen so far quantizes them together, and
#: because the planner never moves them independently.
GEMM_FAMILY_ATTN_GDN = "attn_gdn"
#: Embedding / lm_head.
GEMM_FAMILY_VOCAB = "vocab"
#: Routed and shared experts.
GEMM_FAMILY_MOE = "moe"

#: Every family the score derivation can resolve separately, in log order.
GEMM_FAMILIES: Tuple[str, ...] = (
    GEMM_FAMILY_MLP,
    GEMM_FAMILY_ATTN_GDN,
    GEMM_FAMILY_VOCAB,
    GEMM_FAMILY_MOE,
)

#: Module-path markers per family, tested in THIS order. Order is load-bearing
#: twice over: routed experts live UNDER ``mlp.experts`` so "moe" must be
#: decided before "mlp", and a fused ``qkv_proj`` sits under ``self_attn`` so
#: the attention marker must not be reached by an MLP path first.
_GEMM_FAMILY_MARKERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        GEMM_FAMILY_VOCAB,
        ("lm_head", "embed_tokens", "word_embeddings", "output_layer"),
    ),
    (
        GEMM_FAMILY_MOE,
        ("experts", "shared_expert", "block_sparse_moe", "moe_"),
    ),
    (
        GEMM_FAMILY_ATTN_GDN,
        (
            "self_attn",
            "self_attention",
            "linear_attn",
            "attention",
            "qkv_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
        ),
    ),
    (GEMM_FAMILY_MLP, ("mlp", "feed_forward", "feedforward", "ffn")),
)


def gemm_family_of_module(name: str) -> Optional[str]:
    """The compute family a module path belongs to, or ``None`` when no marker
    matches (a norm, a router, a vision-tower module -- nothing the GEMM score
    describes). Matching is on the LOWERCASED path so a ``re:``-prefixed
    compressed-tensors target and a literal ModelOpt layer name both land."""
    low = name.lower()
    for family, markers in _GEMM_FAMILY_MARKERS:
        if any(marker in low for marker in markers):
            return family
    return None


def _modelopt_algo_format(algo: str) -> Optional[str]:
    """A ModelOpt ``quant_algo`` string -> the ``_FORMAT_LANES`` key it runs
    under, or ``None`` when the algo is one this table has no opinion about
    (the caller then keeps the checkpoint-wide key, never invents a lane)."""
    a = (algo or "").strip().upper()
    if not a or a in ("NONE", "NULL", "BF16", "FP16", "NO_QUANT"):
        return "bf16"
    if a in ("W4A16_NVFP4", "NVFP4_A16", "W4A16_AWQ_NVFP4"):
        return "nvfp4_a16"
    if "NVFP4" in a or a in ("FP4", "W4A4"):
        # NVFP4 / NVFP4_AWQ / W4A4_NVFP4: activations are 4-bit too, so the
        # native block-scaled lane is reachable where the backend resolves to
        # one. The W4A16 spellings above are already excluded.
        return "nvfp4_a4"
    if a.startswith("FP8") or a.startswith("W8A8_FP8"):
        return "fp8"
    if a.startswith("INT4") or a.startswith("W4A16"):
        return "int4"  # no lane table: takes the loud bf16 fallback
    return None


def _ct_group_format(group: dict) -> Optional[str]:
    """The ``_FORMAT_LANES`` key a compressed-tensors ``config_groups`` entry
    runs under, read from its weight/activation bit widths and types."""
    weights = (group or {}).get("weights") or {}
    acts = (group or {}).get("input_activations") or {}
    try:
        bits = int(weights.get("num_bits") or 0)
    except (TypeError, ValueError):
        return None
    wtype = str(weights.get("type") or "").lower()
    if bits == 8 and wtype == "float":
        return "fp8"
    if bits == 4 and wtype == "float":
        try:
            act_bits = int(acts.get("num_bits") or 0)
        except (TypeError, ValueError):
            act_bits = 0
        return "nvfp4_a4" if act_bits == 4 else "nvfp4_a16"
    if bits == 4:
        return "int4"  # no lane table (loud fallback)
    if bits == 8:
        # INT8: only a W8A8 group takes the int8_scaled_mm lane. A weight-only
        # group (no 8-bit input_activations) runs through Marlin wNa16 and has
        # no lane table, so it keeps a key that resolves to the loud bf16
        # fallback rather than borrowing the wrong kernel's number.
        try:
            act_bits = int(acts.get("num_bits") or 0)
        except (TypeError, ValueError):
            act_bits = 0
        return "int8" if act_bits == 8 else "int8_a16"
    return None


def _dominant(values: Sequence[str]) -> Optional[str]:
    """The most frequent entry, ties broken by first appearance (so the result
    does not depend on dict iteration luck)."""
    best, best_count = None, 0
    for value in values:
        count = sum(1 for v in values if v == value)
        if count > best_count:
            best, best_count = value, count
    return best


#: Every family the GEMM score describes, in marker order. A class selector
#: quantizes a COMPLEMENT over this set, so it has to be derivable rather than
#: spelled out a second time.
_ALL_GEMM_FAMILIES: Tuple[str, ...] = tuple(fam for fam, _ in _GEMM_FAMILY_MARKERS)


def _is_class_selector(target: str) -> bool:
    """True for a compressed-tensors target naming a module CLASS, not a path.

    ``targets: ["Linear"]`` is the common shape and it means "every Linear in
    the model", i.e. every GEMM family. A path (``model.layers.0.mlp``) or a
    regex (``re:.*mlp.*``) names specific modules and must NOT be read this
    way -- treating one as a class would quantize families it never mentioned.
    """
    t = (target or "").strip()
    if not t or t.startswith("re:") or "." in t or "*" in t or "/" in t:
        return False
    return t.isidentifier() and gemm_family_of_module(t) is None


def _per_family_formats(
    qc: dict, layer_split: Optional[Mapping[str, int]] = None
) -> Dict[str, str]:
    """``{family: format key}`` for the families the config declares a scheme
    for, ``{}`` when it declares no per-module split at all.

    Three config shapes carry a genuine per-module split:

    * ModelOpt ``MIXED_PRECISION`` (``quantized_layers`` -> per-module algo);
    * compressed-tensors ``config_groups``, at ANY group count -- the split may
      live in ``ignore`` rather than in a second group (#485 item 1);
    * the ``ignore`` list itself: a GEMM family the quantizer was told to skip
      is BF16-RESIDENT, and that is a fact about the checkpoint exactly as much
      as a declared scheme is.

    ``layer_split`` optionally carries #371's per-layer counts
    (``{"gdn": n, "full": n}``). It is used only to resolve the attention
    family, which spans ``self_attn`` AND ``linear_attn`` and therefore has no
    single answer at family granularity -- see the resolution note below.
    """
    per_family: Dict[str, List[Tuple[str, str]]] = {}

    layers = qc.get("quantized_layers")
    if isinstance(layers, dict) and layers:
        for module, info in layers.items():
            family = gemm_family_of_module(str(module))
            if family is None:
                continue
            algo = str((info or {}).get("quant_algo") or "")
            key = _modelopt_algo_format(algo)
            if key is not None:
                per_family.setdefault(family, []).append((key, str(module)))

    # The ignored families first: a class selector's complement is defined
    # against them, so they have to be known before the groups are read.
    ignored: Dict[str, List[str]] = {}
    for entry in qc.get("ignore") or []:
        family = gemm_family_of_module(str(entry))
        if family is not None:
            ignored.setdefault(family, []).append(str(entry))

    groups = qc.get("config_groups")
    if not per_family and isinstance(groups, dict) and groups:
        # NOT `len(groups) > 1` any more (#485 item 1): a single-group config
        # can still describe a split, because the split may live in `ignore`.
        for group in groups.values():
            key = _ct_group_format(group or {})
            if key is None:
                continue
            for target in (group or {}).get("targets") or []:
                target = str(target)
                if _is_class_selector(target):
                    # COMPLEMENT, not enumeration: a class selector covers
                    # every GEMM family the ignore list does not name. A family
                    # it does name is bf16-resident as far as this config can
                    # say, and claiming it here would overwrite that with a
                    # scheme the quantizer was explicitly told not to apply.
                    for family in _ALL_GEMM_FAMILIES:
                        if family not in ignored:
                            per_family.setdefault(family, []).append((key, target))
                    continue
                family = gemm_family_of_module(target)
                if family is not None:
                    per_family.setdefault(family, []).append((key, target))

    for family, entries in ignored.items():
        for entry in entries:
            per_family.setdefault(family, []).append(("bf16", entry))

    resolved: Dict[str, str] = {}
    for family, evidence in per_family.items():
        keys = [key for key, _src in evidence]
        distinct = set(keys)
        if len(distinct) > 1 and "bf16" in distinct:
            # A family carrying both bf16 and quantized evidence is genuinely
            # MIXED. `_dominant` would pick by CONFIG-ENTRY count, which bears
            # no relation to how many LAYERS carry each scheme, so it would
            # invent the answer.
            #
            # For the attention family that split IS knowable: it spans
            # `self_attn` (full-attention layers) and `linear_attn` (GDN
            # layers), and #371's census counts both. Weighted by layers the
            # answer is a measurement rather than a vote. Any other family
            # stays unresolved and is left OUT.
            weighted = _weigh_attn_gdn_by_layers(family, evidence, layer_split)
            if weighted is not None:
                resolved[family] = weighted
            continue
        key = _dominant(keys)
        if key is not None:
            resolved[family] = key
    return resolved


def _weigh_attn_gdn_by_layers(
    family: str,
    evidence: Sequence[Tuple[str, str]],
    layer_split: Optional[Mapping[str, int]],
) -> Optional[str]:
    """Resolve the attention family's mixed evidence by LAYER count (#371).

    Returns ``None`` when the split is unknown or the family is not the
    attention one -- an unresolvable family is omitted rather than guessed.
    """
    if family != GEMM_FAMILY_ATTN_GDN or not layer_split:
        return None
    gdn = int(layer_split.get("gdn", 0) or 0)
    full = int(layer_split.get("full", 0) or 0)
    if gdn <= 0 and full <= 0:
        return None
    weights: Dict[str, int] = {}
    for key, source in evidence:
        # `linear_attn` evidence describes the GDN layers; everything else in
        # this family (self_attn, qkv/o projections) describes the full ones.
        n = gdn if "linear_attn" in source.lower() else full
        weights[key] = weights.get(key, 0) + n
    if not weights:
        return None
    best = max(weights.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else None


def _is_nvfp4_like(method: str, fmt: str, qc: dict) -> Optional[str]:
    """``"nvfp4_a4"`` (activations 4-bit too, so the native FP4 lane is
    reachable), ``"nvfp4_a16"`` (weight-only -> Marlin on EVERY architecture,
    Blackwell included), or ``None``.

    The distinction is not cosmetic and not inferable from the repo name: for
    W4A16_NVFP4 the fork's ``ModelOptNvFp4A16LinearMethod`` is unconditionally
    Marlin, so a 5090 scored on a native-FP4 number would be scored ~2.6x too
    fast."""
    if method in ("modelopt", "modelopt_fp4"):
        algo = str(qc.get("quant_algo") or "").upper()
        if algo == "MIXED_PRECISION":
            # Stopgap for the checkpoint-WIDE key only: the FLOP-dominant
            # family's format. The per-family vectors below are the actual
            # answer; this keeps the scalar consumers on the format most of
            # the model runs in rather than on a bf16 stand-in.
            families = _per_family_formats(qc)
            for family in (GEMM_FAMILY_MOE, GEMM_FAMILY_MLP):
                if family in families:
                    return families[family]
            return _dominant(list(families.values()))
        return _modelopt_algo_format(algo)
    if "nvfp4" in fmt:  # nvfp4-pack-quantized (compressed-tensors)
        for group in (qc.get("config_groups") or {}).values():
            key = _ct_group_format(group or {})
            if key in ("nvfp4_a4", "nvfp4_a16"):
                return key
    return None


def _checkpoint_quant_config(model_path: Optional[str], cfg: dict) -> dict:
    """The quantization block, from ``config.json`` or -- for a ModelOpt export
    that keeps it out of the model config -- from ``hf_quant_config.json``,
    whose payload sits under a ``quantization`` key (both shapes are what
    ``ModelOptMixedPrecisionConfig.from_config`` accepts).

    Read only by the compute-format dispatch. The byte model keeps reading
    ``_quant_config`` alone so its sizing is unchanged."""
    qc = _quant_config(cfg)
    if qc or not model_path or _is_gguf_model(model_path):
        return qc
    try:
        with open(os.path.join(model_path, "hf_quant_config.json")) as f:
            payload = json.load(f)
    except Exception:
        return {}
    section = payload.get("quantization")
    return section if isinstance(section, dict) else (payload or {})


def checkpoint_compute_format(model_path: Optional[str]) -> Tuple[str, str]:
    """``(format key, one-line description)`` for the checkpoint's weight format.

    The key indexes ``_FORMAT_LANES``. ``"bf16"`` for an unquantized
    checkpoint (its lane IS the dense probe, so the default path is unchanged);
    ``"fp8"`` for fp8-like schemes; otherwise the scheme's own name, which has
    no lane table and therefore takes the warned bf16 fallback.

    Read from the checkpoint's own config, never from the repo or path name:
    a directory called ``...-FP8`` that ships int4 weights would otherwise be
    scored on a lane it never runs."""
    if _is_gguf_model(model_path):
        return "gguf", "GGUF (ggml k-quant / legacy types)"
    try:
        cfg = _load_checkpoint_config(model_path)
    except Exception as ex:
        return (
            "unknown",
            f"unreadable checkpoint config ({type(ex).__name__}: {ex})",
        )
    qc = _checkpoint_quant_config(model_path, cfg)
    if not qc:
        return "bf16", "unquantized (no quantization_config)"
    method = str(qc.get("quant_method") or "").lower()
    fmt = str(qc.get("format") or qc.get("fmt") or "").lower()
    algo = str(qc.get("quant_algo") or "").upper()
    if not method and algo:
        # hf_quant_config.json shape: the algo IS the method declaration.
        method = "modelopt"
    nvfp4 = _is_nvfp4_like(method, fmt, qc)
    if nvfp4 is not None:
        detail = f"quant_algo={algo or '?'}" if algo else f"fmt={fmt or '?'}"
        return nvfp4, f"{nvfp4} (quant_method={method or '?'}, {detail})"
    if _is_fp8_like(method, fmt):
        block = qc.get("weight_block_size")
        shape = f", weight_block_size {block}" if block else ""
        return "fp8", f"fp8 (quant_method={method or '?'}, fmt={fmt or '?'}{shape})"
    if _is_int8_w8a8_like(method, fmt, qc):
        return (
            "int8",
            f"int8 W8A8 (quant_method={method or '?'}, fmt={fmt or '?'}, "
            "dynamic per-token activations)",
        )
    return (
        method or "unknown",
        f"quant_method={method or '?'}, format={fmt or '?'}",
    )


def checkpoint_compute_format_families(
    model_path: Optional[str],
) -> Tuple[str, str, Dict[str, str]]:
    """``(checkpoint-wide format key, description, per-family format keys)``.

    The third element is EMPTY for every checkpoint that applies ONE scheme to
    every family -- which is every checkpoint the scalar path was written for.
    It is populated only when the config genuinely declares different schemes
    for different families, so a caller that ignores it, or a family missing
    from it, keeps reading exactly the number it read before. That is the whole
    migration story: there is no new profile field and no new file format, only
    a dict that is empty until a mixed-precision checkpoint fills it."""
    fmt, desc = checkpoint_compute_format(model_path)
    if _is_gguf_model(model_path):
        return fmt, desc, {}
    try:
        cfg = _load_checkpoint_config(model_path)
    except Exception:
        return fmt, desc, {}
    qc = _checkpoint_quant_config(model_path, cfg)
    if not qc:
        return fmt, desc, {}
    # #371's per-layer counts, so a family that spans two layer kinds can be
    # resolved by measurement instead of by config-entry vote. Same derivation
    # the checkpoint model uses (`layer_types` -> full / linear attention).
    text = cfg.get("text_config", cfg) if isinstance(cfg, dict) else {}
    layer_types = (text or {}).get("layer_types") or []
    layer_split = {
        "full": sum(1 for t in layer_types if t == "full_attention"),
        "gdn": sum(1 for t in layer_types if t == "linear_attention"),
    }

    families = _per_family_formats(qc, layer_split=layer_split or None)
    if len(set(families.values())) <= 1:
        # One scheme across the board (or none declared per module): the
        # checkpoint-wide key already says everything the families would.
        return fmt, desc, {}
    detail = ", ".join(f"{fam}={families[fam]}" for fam in sorted(families))
    if layer_split.get("gdn") and GEMM_FAMILY_ATTN_GDN in families:
        # The attention family spans two layer kinds, so a single key is a
        # majority statement rather than a uniform one. Say so, and say which
        # layers the key does NOT describe -- a reader who assumes uniformity
        # here would mis-price the minority.
        gdn, full = layer_split["gdn"], layer_split["full"]
        key = families[GEMM_FAMILY_ATTN_GDN]
        minority = (
            f", and does NOT describe the {full} full_attention layer(s)"
            if key == "bf16" and full
            else f", and does NOT describe the {gdn} linear_attention layer(s)"
            if key != "bf16" and gdn
            else ""
        )
        detail += (
            f" [attn_gdn spans {gdn} linear_attention / {full} full_attention"
            f" layers; the key describes its majority{minority}]"
        )
    return fmt, f"{desc}; per-family: {detail}", families


def rank_gemm_scores(
    entries: Sequence[dict], fmt: str
) -> Tuple[List[float], List[str], List[str]]:
    """Per-rank compute score in the checkpoint's OWN format.

    Returns ``(scores, lane labels, warnings)``. For each rank the first lane
    of ``_FORMAT_LANES[fmt]`` that the profile has a measurement for on that
    card wins, mirroring the order the serving path tries them. Cards on the
    same rig can land on different lanes -- that is the point: an fp8
    checkpoint runs on sm_120 tensor cores on one card and through a
    weight-only lane on the next, and the ratio between those two is what the
    prefill objective is made of.

    Two fallbacks, both LOUD, because the failure they guard against is a bf16
    number silently wearing a quantized label:

    * the format has no lane table (AWQ / GPTQ / GGUF today);
    * the format has one but the profile measured none of its lanes on this
      card (a profile written before the lanes existed, or every lane refused
      on that architecture).

    Both fall back to the dense bf16 score, which is the pre-existing
    behaviour, so the fallback path is a warning about accuracy and never a
    behaviour change."""
    lanes = _FORMAT_LANES.get(fmt)
    scores: List[float] = []
    labels: List[str] = []
    warnings: List[str] = []
    if lanes is None:
        for entry in entries:
            scores.append(float(entry["gemm_tflops"]))
            labels.append(_LANE_LABELS[LANE_BF16])
        known = FORMATS_WITHOUT_LANES.get(fmt)
        if known:
            warnings.append(
                f"checkpoint format '{fmt}' is recognised but has no measured "
                f"GEMM lane: {known}. The prefill objective is scoring every "
                "rank on the DENSE BF16 probe, which is the wrong format for "
                "this checkpoint; this is a KNOWN gap, not an unrecognised "
                "format."
            )
            for entry in entries:
                scores.append(float(entry["gemm_tflops"]))
                labels.append(_LANE_LABELS[LANE_BF16])
            return scores, labels, warnings
        warnings.append(
            f"checkpoint format '{fmt}' has no GEMM lane table: the prefill "
            "objective is scoring every rank on the DENSE BF16 probe, which "
            "is the wrong format for this checkpoint and compresses the "
            "card-to-card compute ratio (measured on the reference rig: 3.79 "
            "in bf16 against 8.64 for the same cards on an fp8 checkpoint). "
            "The chosen MLP vector is a lower bound on the concentration this "
            "rig wants; pin --rank-mlp-ratio if you have measured better."
        )
        return scores, labels, warnings

    for entry in entries:
        # The dense lane is always available -- it IS ``gemm_tflops``, probed
        # unconditionally -- so it is not carried in the per-card lane map.
        available = dict(entry.get("gemm_lanes") or {})
        available[LANE_BF16] = float(entry["gemm_tflops"])
        chosen = next((lane for lane in lanes if lane in available), None)
        if chosen is None:
            scores.append(float(entry["gemm_tflops"]))
            labels.append(_LANE_LABELS[LANE_BF16] + " (fallback)")
            notes = entry.get("gemm_lane_notes") or {}
            detail = "; ".join(
                f"{lane}: {notes.get(lane, 'not probed')}" for lane in lanes
            )
            # Do not tell the reader to re-probe a lane that has no probe: a
            # format whose lanes are registered for DISPATCH ORDER only cannot
            # be measured yet, however often the profile is rebuilt.
            hint = (
                "This format's lanes have no probe yet, so re-probing cannot "
                "produce one."
                if all(lane not in _LANE_PROBES for lane in lanes)
                else "Re-run with SGLANG_PERF_REPROBE=1 if the profile "
                "predates the lane probes."
            )
            warnings.append(
                f"{entry.get('name', 'GPU')}: no {fmt} GEMM lane measured on "
                f"this card ({detail}) -- scoring it on the DENSE BF16 probe "
                f"instead. {hint} Otherwise this rank's compute "
                "score is in the wrong format and the ratio it forms with the "
                "other ranks is understated."
            )
        else:
            scores.append(float(available[chosen]))
            labels.append(_LANE_LABELS.get(chosen, chosen))
    return scores, labels, warnings


@dataclasses.dataclass
class GemmScores:
    """Per-rank compute scores widened by one axis: the compute family.

    ``scalar`` is the pre-existing per-rank vector, unchanged and still the
    answer for every consumer that has no family of its own. ``families`` holds
    a vector ONLY for a family whose format differs from the checkpoint-wide
    one, so on a uniform checkpoint it is empty and every lookup returns the
    same floats the scalar path returned -- not an equal value, the same
    object's contents.
    """

    #: Per-rank score in the checkpoint-wide format.
    scalar: List[float]
    #: Lane label per rank for ``scalar``.
    scalar_labels: List[str]
    #: family -> per-rank score, only for families that diverge.
    families: Dict[str, List[float]]
    #: family -> lane label per rank, same keys as ``families``.
    family_labels: Dict[str, List[str]]
    #: family -> format key, as read from the checkpoint config.
    family_formats: Dict[str, str]
    #: Loud fallbacks, family-tagged where they came from a family lookup.
    warnings: List[str]

    @property
    def mixed(self) -> bool:
        """Whether any family genuinely scores differently from the scalar."""
        return bool(self.families)

    def for_family(self, family: str) -> List[float]:
        """This family's per-rank scores, or the scalar when it has none."""
        return list(self.families.get(family) or self.scalar)

    def resolve(self, *preferred: str) -> Tuple[List[float], str]:
        """``(scores, source name)`` for the first preferred family that has
        its own vector, else ``(scalar, "scalar")``.

        The source name exists so the plan log can NAME the fallback instead of
        printing a vector whose provenance the reader has to guess."""
        for family in preferred:
            if family in self.families:
                return list(self.families[family]), family
        return list(self.scalar), "scalar"


def rank_gemm_family_scores(
    entries: Sequence[dict],
    fmt: str,
    family_formats: Optional[Dict[str, str]] = None,
) -> GemmScores:
    """``rank_gemm_scores`` widened to (rank, family).

    ``family_formats`` is what ``checkpoint_compute_format_families`` returned:
    empty for every checkpoint with one scheme, in which case this is the
    scalar call and nothing else. A family whose format equals the
    checkpoint-wide one is deliberately NOT stored -- it would be the same
    numbers under a second key, and an empty ``families`` is what makes
    ``mixed`` mean what it says.

    Each family resolves through the SAME per-card lane order as the scalar
    path, so two families on one card can land on two different lanes exactly
    when the serving path would put them there (a W4A16 family on Marlin while
    an fp8 family takes the native tensor path on the very same 5090)."""
    scalar, scalar_labels, warnings = rank_gemm_scores(entries, fmt)
    families: Dict[str, List[float]] = {}
    family_labels: Dict[str, List[str]] = {}
    by_format: Dict[str, Tuple[List[float], List[str]]] = {}
    for family in GEMM_FAMILIES:
        family_fmt = (family_formats or {}).get(family)
        if family_fmt is None or family_fmt == fmt:
            continue
        if family_fmt not in by_format:
            scores, labels, warns = rank_gemm_scores(entries, family_fmt)
            by_format[family_fmt] = (scores, labels)
            warnings.extend(f"[{family_fmt}] {w}" for w in warns)
        scores, labels = by_format[family_fmt]
        families[family] = list(scores)
        family_labels[family] = list(labels)
    return GemmScores(
        scalar=scalar,
        scalar_labels=scalar_labels,
        families=families,
        family_labels=family_labels,
        family_formats=dict(family_formats or {}),
        warnings=warnings,
    )


#: Cost-model weight family (``PerfCostModel.families`` keys) -> GEMM score
#: family (#324 ``GemmScores`` axis). ``None`` = no family of its own, the
#: scalar (checkpoint-wide) rate applies. The MLP entries are resolved at
#: call time because the mlp mass IS the routed-expert mass on a MoE
#: checkpoint (see ``_build_families``), where the ``moe`` score family
#: carries it.
_WEIGHT_TO_GEMM_FAMILY: Dict[str, Optional[str]] = {
    "attn": GEMM_FAMILY_ATTN_GDN,
    "gdn": GEMM_FAMILY_ATTN_GDN,
    "draft_attn": GEMM_FAMILY_ATTN_GDN,
    "vocab": GEMM_FAMILY_VOCAB,
    # vision towers ship unquantized (bf16) in every checkpoint the byte
    # model has seen; no measured family lane exists for them, so the
    # honest rate is the scalar.
    "vision": None,
    "draft_repl": None,  # replicated: skipped by the sharded term anyway
    "draft_solo_ckpt": None,  # external checkpoint bytes, format unknown
}


def gemm_family_for_weight_family(name: str, num_experts: int) -> Optional[str]:
    """The #324 score family a cost-model weight family runs on, or ``None``
    for 'no own family, use the scalar rate'."""
    if name in ("mlp", "draft_mlp"):
        return GEMM_FAMILY_MOE if num_experts > 0 else GEMM_FAMILY_MLP
    return _WEIGHT_TO_GEMM_FAMILY.get(name)


def family_prefill_tflops(
    model: PerfCostModel, gemm: GemmScores
) -> Optional[Dict[str, List[float]]]:
    """Per-family compute rates for the ``sum(p/r)`` prefill arithmetic
    (#287, deliberately deferred by #324 to keep that merge byte-identical).

    ``None`` unless the checkpoint is genuinely MIXED: with an empty
    ``gemm.families`` every family would map to the scalar vector and the
    arithmetic, while mathematically equal, would differ from the scalar
    path in the last float bits -- returning ``None`` keeps every
    single-scheme plan bit-for-bit unchanged. On a mixed checkpoint each
    weight family gets the per-rank vector of the lane its own format
    resolves to (``GemmScores.for_family``: the family's own vector when one
    diverges, the scalar otherwise).
    """
    if not gemm.mixed:
        return None
    out: Dict[str, List[float]] = {}
    for name in model.families:
        family = gemm_family_for_weight_family(name, model.num_experts)
        out[name] = (
            list(gemm.for_family(family)) if family else list(gemm.scalar)
        )
    return out


def effective_prefill_tflops(
    model: PerfCostModel, gemm: GemmScores
) -> List[float]:
    """Per-rank EFFECTIVE prefill compute rate, family-blended by the
    ``sum(p/r)`` arithmetic at the base plan: ``sum_fam(p_fam) /
    sum_fam(p_fam / r_fam)`` -- the harmonic, mass-weighted blend of the
    lane rates each family actually runs on that rank.

    On a single-scheme checkpoint this is exactly ``gemm.scalar`` (every
    family divides by the same rate, the blend collapses algebraically and
    the scalar is returned as the SAME list content, no arithmetic run).
    Consumers: the #287 depth/format-aware operating grid, which needs one
    per-card rate but must not lose the family axis."""
    fam_map = family_prefill_tflops(model, gemm)
    if fam_map is None:
        return list(gemm.scalar)
    out: List[float] = []
    for r in range(model.tp_size):
        params = 0.0
        time = 0.0
        for name, fam in model.families.items():
            if fam.params <= 0 or fam.shard == "replicated":
                continue
            p_r = fam.params * model._shard_fractions(fam.shard, model.base_plan)[r]
            rates = fam_map.get(name)
            rate_r = rates[r] if rates else gemm.scalar[r]
            params += p_r
            time += p_r / rate_r
        out.append(params / time if time > 0 else gemm.scalar[r])
    return out


def vocab_ratio_from_membw(membw_gbs: Sequence[float], base: int = 6) -> List[int]:
    """Integer weight vector proportional to the per-rank memory-bandwidth
    scores, for ratio-weighted vocab sharding (--rank-vocab-ratio auto).

    The lm_head matvec is bandwidth-bound (it streams the whole vocab shard
    once per forward), so shard widths proportional to membw equalize the
    per-rank read TIME. Scaled so the smallest rank gets `base` units
    (granularity ~ base/sum, ample for the 64-row padded vocab units), then
    gcd-reduced. Example: 1558/723/723 GB/s -> [13, 6, 6]."""
    low = min(membw_gbs)
    assert low > 0, f"non-positive membw scores: {membw_gbs}"
    scaled = [max(1, round(b / low * base)) for b in membw_gbs]
    g = math.gcd(*scaled)
    return [s // g for s in scaled]


# ---------------------------------------------------------------------------
# Measured KV-budget registry (shared fingerprint).
#
# The registry file (written post-capture by
# model_runner_kv_cache_mixin.note_post_capture_leftover, read pre-boot by
# apply_auto_performance) is keyed by a config fingerprint. Writer and reader
# MUST agree on the fields byte-for-byte, so the fingerprint lives here (a
# stdlib-only module both can import; the mixin pulls torch). Deliberately
# EXCLUDED from the fields: rank_mlp_ratio / the chosen weight vector — the
# whole point of the pre-boot weight planner is to move weights BETWEEN boots
# of the same configuration, and re-keying on the vector would discard the
# measured residency the planner needs to choose it.
# ---------------------------------------------------------------------------


def measured_kv_budget_fingerprint_fields(server_args) -> dict:
    sa = server_args
    fields = {
        "model_path": sa.model_path,
        "tp_size": sa.tp_size,
        "rank_gpu_id": getattr(sa, "rank_gpu_id", None),
        "rank_tp_ratio": getattr(sa, "rank_tp_ratio", None),
        "rank_kv_ratio": getattr(sa, "rank_kv_ratio", None),
        "rank_auto_reserve_mib": getattr(sa, "rank_auto_reserve_mib", None),
        "rank_gpu_memory_mib": getattr(sa, "rank_gpu_memory_mib", None),
        "mem_fraction_static": sa.mem_fraction_static,
        "kv_cache_dtype": sa.kv_cache_dtype,
        "context_length": sa.context_length,
        "page_size": sa.page_size,
        "quantization": sa.quantization,
        "max_running_requests": sa.max_running_requests,
        "chunked_prefill_size": sa.chunked_prefill_size,
        "spec_algorithm": sa.speculative_algorithm,
        "spec_draft_model": sa.speculative_draft_model_path,
        "spec_cross": getattr(sa, "speculative_cross_algorithm", False),
        "spec_cross_force": getattr(
            sa, "speculative_cross_algorithm_force", None
        ),
        "spec_adaptive": sa.speculative_adaptive,
        "spec_adaptive_config": sa.speculative_adaptive_config,
        # RAW draft-token count, deliberately NOT max_speculative_num_draft_
        # tokens: that is a cached_property which resolves the cross-rung
        # shapes — evaluating it at parse time (before the speculative hook
        # runs) caches the WRONG value (4 instead of 16 on the T156 rig) and
        # under-sizes the shared logits buffer at graph capture (measured
        # 2026-07-22: assert 'holds 8 rows but caller needs 32'). The raw
        # field + spec_adaptive_config + spec_cross_force carry the same
        # config identity.
        "spec_max_draft_tokens": sa.speculative_num_draft_tokens,
        "cuda_graph_max_bs": getattr(
            sa.cuda_graph_config.decode, "max_bs", None
        ),
    }
    # force=policy: an EXPLICIT drafter-policy table decides which NEXTN k
    # states get built (and, via the T156-D ctx cap, the DFLASH solo pool
    # size), so different tables have different boot footprints and their
    # corrections are not transferable (the role spec_adaptive_config plays
    # for the auto mode). Included ONLY when set: the unset/auto default is
    # a pure function of already-fingerprinted fields (drafter config +
    # force), and omitting it keeps every pre-existing registry digest
    # valid (adding the key unconditionally would orphan all of them).
    drafter_policy = getattr(sa, "speculative_drafter_policy", None)
    if drafter_policy is not None:
        fields["spec_drafter_policy"] = drafter_policy
    # #201 slice 3: a pipeline's stages have different footprints (layer
    # windows, embed/lm_head asymmetry), so a record measured under one
    # pp geometry must not be replayed under another. Included ONLY when a
    # pipeline is configured -- same rationale as spec_drafter_policy
    # above: pp_size == 1 keeps every pre-existing digest valid. The
    # per-STAGE identity is a runner-side path suffix
    # (model_runner_kv_cache_mixin._measured_kv_budget_cache_path), because
    # this fingerprint is parse-time and has no pp_rank.
    pp_size = getattr(sa, "pp_size", 1) or 1
    if pp_size > 1:
        fields["pp_size"] = pp_size
        fields["pp_layer_ratio"] = getattr(sa, "pp_layer_ratio", None)
    return fields


def measured_kv_budget_cache_path(server_args) -> str:
    # Timing subtlety: the pre-boot weight planner runs EARLY in ServerArgs
    # __post_init__ (before e.g. mem_fraction_static is defaulted), the
    # registry writer runs at boot on the fully resolved args — the same
    # fields would hash differently. The planner therefore stashes its
    # computed path on the args object (pickled through to the scheduler
    # processes), and every later call returns the stash: reader and writer
    # agree by construction, and the stash is boot-stable because parse-time
    # resolution is a pure function of the CLI.
    stashed = getattr(server_args, "_measured_kv_budget_registry_path", None)
    if stashed:
        return stashed
    fields = measured_kv_budget_fingerprint_fields(server_args)
    digest = hashlib.sha1(
        json.dumps(fields, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    os.makedirs(PROFILE_CACHE_DIR, exist_ok=True)
    return os.path.join(PROFILE_CACHE_DIR, f"kv_budget-{digest}.json")


def load_measured_registry(server_args) -> Optional[dict]:
    """The measured KV-budget registry with a complete component balance.

    Returns the registry dict (keys: ``components`` — one dict per TP rank,
    see note_post_capture_leftover for the schema — and ``mlp_vector``, the
    weight vector the measurement was taken under) when the file exists,
    every rank's balance is complete, and the measured-budget mode is
    enabled; else None. The planner treats an incomplete registry exactly
    like a first boot: fall back to the static heuristics, measure, converge
    on the next boot."""
    try:
        from sglang.srt.environ import envs

        if not envs.SGLANG_MEASURED_KV_BUDGET.get():
            return None
    except Exception:  # pragma: no cover - envs import is boot-critical only
        return None
    try:
        with open(measured_kv_budget_cache_path(server_args)) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    comps = data.get("components")
    if (
        not isinstance(comps, list)
        or len(comps) != server_args.tp_size
        or not all(isinstance(c, dict) and c for c in comps)
    ):
        return None
    required = (
        "device_total_bytes",
        "ranks_on_gpu",
        "residual_residency_bytes",
        "weights_alloc_bytes",
        "required_free_bytes",
        "mamba_aux_pool_bytes",
    )
    for c in comps:
        if any(k not in c for k in required):
            return None
    vec = data.get("mlp_vector")
    if not isinstance(vec, list) or len(vec) != server_args.tp_size:
        return None
    if not measured_registry_cards_still_present(comps):
        return None
    return data


def measured_registry_cards_still_present(components: list) -> bool:
    """Are the cards this balance was measured on the cards present now?

    AUDIT #331. The registry is indexed by TP rank, and a rank is not a
    physical card: between two boots the same rank can land on a different
    GPU (a card pulled, ``--rank-gpu-id`` edited, CUDA re-enumerated), and the
    stored ``device_total_bytes`` / residency balance then describes hardware
    that is not there. Each component therefore carries the ``card_uuid`` it
    was measured on, and a registry naming a card this host no longer has is
    discarded rather than replayed -- a first boot re-measures in one run,
    while a wrong total mis-sizes the KV pool silently.

    Two deliberate non-failures. A component written before #331 has no
    ``card_uuid``; it is accepted with a warning and re-stamped by the next
    post-capture write, because rejecting every pre-existing registry would
    throw away real measured convergence to guard against a card change that
    probably did not happen. And an unreachable NVML accepts as well: the
    check exists to catch a changed rig, not to make a driver hiccup lose the
    registry.
    """
    stored = [
        c.get("card_uuid") for c in components if isinstance(c, dict) and c.get("card_uuid")
    ]
    if not stored:
        logger.warning(
            "Measured KV-budget: the stored registry predates card-identity "
            "stamping (#331), so which physical cards it was measured on "
            "cannot be verified. Using it; the next post-capture write stamps it."
        )
        return True
    try:
        from sglang.srt.registry import nvml

        if not nvml.is_available():
            return True
        # list_devices, not identity_map: this runs at parse time in the
        # launcher, and the CUDA half of the map would create a context in
        # the process that is about to fork its workers. Presence is a
        # question about NVML alone.
        present = {d.uuid for d in nvml.list_devices()}
    except Exception as exc:  # noqa: BLE001 - never lose the registry to NVML
        logger.debug("Measured KV-budget: card presence check skipped (%s)", exc)
        return True
    missing = sorted({u for u in stored if u not in present})
    if missing:
        logger.warning(
            "Measured KV-budget: the stored registry was measured on card(s) "
            "%s, which this host does not report any more (present: %s). "
            "Discarding it and re-measuring rather than sizing the KV pool "
            "against another card's total (#331).",
            ", ".join(missing),
            ", ".join(sorted(present)) or "none",
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Cost model: per-rank weight bytes + capacity prediction from the model
# config, mirroring the terms the real pool sizing pays (M22 cost-model
# musts: SSM pool moves with GDN units x concurrency, BF16 families inside
# INT4 checkpoints, spec-decode draft weights [embed/lm_head dupes are shared
# BEFORE profiling since eb764a12b, so only the draft's own layer shards and
# fc remain], graph/activation reserves).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PlanInputs:
    """The small, boot-free input contract of the parse-time cost model.

    ``PerfCostModel`` (and the offline planner, ``sglang.srt.planner``)
    consume this dataclass instead of a full ``ServerArgs`` object, so the
    capacity math is callable as a pure library — the design-#97 "single
    source of truth" guarantee: the boot path builds a ``PlanInputs`` from
    itself (``from_server_args``) and the offline planner builds one from
    CLI/manual inputs, and both run the IDENTICAL sizing code.

    Only stdlib types; importing/constructing this never touches torch,
    CUDA, or NVML.
    """

    # -- model + speculative config (what PerfCostModel reads) --------------
    tp_size: int
    model_path: str
    kv_cache_dtype: str = "auto"
    speculative_algorithm: Optional[str] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_draft_model_path: Optional[str] = None
    #: --speculative-draft-placement. "split" (default) puts a draft SHARD on
    #: every rank; "solo" puts the WHOLE unsharded draft -- weights, a
    #: globally-sized draft KV pool, and the draft graphs -- on one rank and
    #: nothing on the others. The weight planner must know which, because the
    #: two placements have opposite per-rank cost profiles.
    speculative_draft_placement: str = "split"
    #: Rank hosting the solo draft (ignored unless placement == "solo").
    speculative_draft_solo_rank: int = 0
    #: --speculative-cross-algorithm (T156). Under the cross gate the GLOBAL
    #: placement stays "split" (the NEXTN/MTP rung is split), but the DFLASH
    #: rung's draft — weights, a draft KV pool sized to the GLOBAL context,
    #: and its graph set — is ALWAYS solo-resident on rank 0. The weight
    #: planner must know this: blind to it, it predicted rank 0's capacity
    #: vector-independent (231k tokens vs ~25k real on the T156 rig) and
    #: concentrated the MLP mass on the one rank that structurally cannot
    #: hold it. Mirrors pool_configurator.solo_draft_kv_cell_factor's cross
    #: clause and the mixin's _is_solo_draft_kv_host.
    speculative_cross_algorithm: bool = False
    max_running_requests: Optional[int] = None
    disable_cuda_graph: bool = False
    #: Include the vision tower in the resident weight budget. These are VL
    #: checkpoints; when a rig serves them TEXT-ONLY the vision encoder (the
    #: ``model.visual.*`` blocks + patch-merger for an HF checkpoint, or the
    #: ``mmproj-*.gguf`` sidecar for GGUF) is not loaded, so its bytes free up
    #: for KV cache. Default True = size the full multimodal footprint (matches
    #: the on-disk checkpoint, which ships the vision weights); False sizes the
    #: text-only resident set (smaller weights -> more KV tokens).
    include_vision: bool = True

    # -- placement + per-card budget (design §2.5) ---------------------------
    #: rank -> physical GPU index (duplicates = co-located ranks).
    rank_gpu_id: Optional[List[int]] = None
    #: Per-rank absolute byte budget ceiling in MiB. This is the BUDGETED
    #: value (NVML total minus the user's free-reserve minus the auto
    #: reserve), not the physical maximum — every capacity number derived
    #: from it is "max possible under your set budget".
    effective_vram_mib: Optional[List[int]] = None

    # -- manual overrides (design §2.6); None => auto-derive that knob -------
    rank_tp_ratio: Optional[List[int]] = None
    rank_mlp_ratio: Optional[List[int]] = None
    rank_moe_ratio: Optional[List[int]] = None
    rank_vocab_ratio: Optional[List[int]] = None
    dcp_size: Optional[int] = None
    kv_token_vector: Optional[List[int]] = None

    @property
    def rank_gpu_memory_mib(self):
        """Alias so functions that duck-type a ``ServerArgs`` (e.g.
        ``resolve_cp_token_ratios``) accept a ``PlanInputs`` unchanged."""
        return self.effective_vram_mib

    @classmethod
    def from_server_args(cls, server_args) -> PlanInputs:
        """Build the cost-model inputs from a (post-validation) ServerArgs.

        Called on the boot path right before ``PerfCostModel`` is
        constructed, so server and offline planner share one input shape.
        """
        budgets = getattr(server_args, "rank_gpu_memory_mib", None)
        if isinstance(budgets, int):
            budgets = [budgets] * server_args.tp_size
        ratio = getattr(server_args, "rank_tp_ratio", None)
        return cls(
            tp_size=server_args.tp_size,
            model_path=server_args.model_path,
            kv_cache_dtype=str(server_args.kv_cache_dtype or "auto"),
            speculative_algorithm=server_args.speculative_algorithm,
            speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            speculative_draft_model_path=server_args.speculative_draft_model_path,
            speculative_draft_placement=str(
                getattr(server_args, "speculative_draft_placement", "split")
                or "split"
            ),
            speculative_draft_solo_rank=(
                server_args.speculative_draft_solo_rank()
                if getattr(server_args, "speculative_draft_placement", "split")
                == "solo"
                and hasattr(server_args, "speculative_draft_solo_rank")
                else 0
            ),
            speculative_cross_algorithm=bool(
                getattr(server_args, "speculative_cross_algorithm", False)
            ),
            max_running_requests=server_args.max_running_requests,
            disable_cuda_graph=bool(
                getattr(server_args, "disable_cuda_graph", False)
            ),
            rank_gpu_id=(
                list(server_args.rank_gpu_id)
                if getattr(server_args, "rank_gpu_id", None)
                else None
            ),
            effective_vram_mib=list(budgets) if budgets else None,
            rank_tp_ratio=list(ratio) if isinstance(ratio, list) else None,
            dcp_size=getattr(server_args, "dcp_size", None),
        )


@dataclasses.dataclass(frozen=True)
class LayerFamilyCensus:
    """How many layers actually CARRY each weight family, and how wide (#371).

    The uneven-TP family table used to assume every layer carries every
    family: attention params were ``full_layers * attn_layer`` and MLP params
    ``n_layers * mlp_layer``, with one uniform per-layer size each. A
    hybrid checkpoint corrects the attention count through ``layer_types``,
    but a HETEROGENEOUS-LAYER checkpoint (Nemotron-NAS / Puzzle) declares
    ``block_configs`` instead: per layer, ``attention.no_op`` / ``ffn.no_op``
    say the block is ABSENT, and ``ffn.ffn_mult`` gives that layer its own
    FFN width. Such a config has no ``layer_types``, so it fell through to
    "every layer is a full-attention MLP layer" and the table counted weights
    that do not exist.

    Two consequences, both silent: the weight-byte model over-counts, and the
    unit partition derived from it hands ranks a share of a family that is not
    there. The #324 per-(rank, family) scores and the #348b cost library read
    the same table, so the error propagates into the plan rather than showing
    up as a load error.

    ``ffn_width_factor`` is the SUM over layers of that layer's FFN width
    relative to the config's nominal ``intermediate_size`` -- so a uniform
    stack of N layers gives exactly ``N`` and the arithmetic is unchanged,
    while a variable-width stack is summed instead of multiplied.
    """

    n_layers: int
    attn_layers: int
    ffn_layers: int
    ffn_width_factor: float
    heterogeneous: bool

    @property
    def uniform(self) -> bool:
        return not self.heterogeneous


def layer_family_census(text: dict, n_layers: int) -> LayerFamilyCensus:
    """Read the per-layer family census off a text config (#371).

    Returns the UNIFORM census (every layer carries every family, width
    factor == ``n_layers``) for every checkpoint that does not declare
    ``block_configs`` -- which is every model this fork serves today, so the
    family table below stays byte-identical for them.

    Tolerant by construction: ``block_configs`` entries may be dicts (raw
    JSON) or objects (a parsed HF config), and a malformed or partial entry
    is treated as a PRESENT block. Over-counting a block that is absent is
    the error this function exists to remove, but under-counting one that is
    present would size a pool too small and fail a boot -- so when the shape
    is unreadable the census degrades toward today's behaviour rather than
    toward the smaller number.
    """
    blocks = text.get("block_configs")
    if not blocks:
        return LayerFamilyCensus(
            n_layers=n_layers,
            attn_layers=n_layers,
            ffn_layers=n_layers,
            ffn_width_factor=float(n_layers),
            heterogeneous=False,
        )

    def _sub(block, name):
        if isinstance(block, dict):
            return block.get(name)
        return getattr(block, name, None)

    def _field(sub, name, default=None):
        if sub is None:
            return default
        if isinstance(sub, dict):
            return sub.get(name, default)
        return getattr(sub, name, default)

    nominal_mult = None
    attn_layers = 0
    ffn_layers = 0
    width = 0.0
    for block in blocks:
        attn = _sub(block, "attention")
        ffn = _sub(block, "ffn")
        if not _field(attn, "no_op", False):
            attn_layers += 1
        if not _field(ffn, "no_op", False):
            ffn_layers += 1
            mult = _field(ffn, "ffn_mult", None)
            if mult is None:
                width += 1.0
            else:
                # Widths are relative: the first real FFN sets the reference,
                # so a stack that is uniform in ffn_mult still yields exactly
                # its layer count and the byte model does not move.
                if nominal_mult is None:
                    nominal_mult = float(mult)
                width += (
                    float(mult) / nominal_mult if nominal_mult else 1.0
                )
    n = len(blocks)
    return LayerFamilyCensus(
        n_layers=n,
        attn_layers=attn_layers,
        ffn_layers=ffn_layers,
        ffn_width_factor=width,
        heterogeneous=(attn_layers != n or ffn_layers != n or width != float(n)),
    )


@dataclasses.dataclass
class _Family:
    """One weight family: total parameter count, bytes per parameter, and
    how it shards across ranks ('attn'/'gdn' follow the base plan on their
    unit grid, 'mlp' follows the candidate vector, 'even' splits evenly,
    'replicated' is per-rank constant)."""

    params: float
    bytes_per_param: float
    shard: str  # attn | gdn | mlp | even | replicated

    @property
    def bytes(self) -> float:
        return self.params * self.bytes_per_param


# ---------------------------------------------------------------------------
# GGUF metadata reader (dependency-free; mirrors the header-only reader in
# the fork's GGUF plugin). Needed because for a GGUF checkpoint the model
# path is a single .gguf FILE, so the HF ``config.json`` the cost model would
# otherwise open does not exist next to it (older builds crashed here with
# NotADirectoryError). We read the fields the cost model needs straight from
# the GGUF key/value header, and derive per-weight-family bytes/element from
# the tensor-info block's ggml quant types so the family byte model is
# roughly correct for quantized GGUF checkpoints (embed/lm_head, attention
# and MLP are quantized here, unlike the "BF16-inside-INT4" safetensors
# assumption). Duplicating ~30 lines of header parsing keeps this module
# self-contained (no import from the model loader) and the scope clean.
# ---------------------------------------------------------------------------

#: ggml quant type -> (block_size, bytes_per_block). bytes/element =
#: bytes_per_block / block_size. Covers the k-quant / legacy / IQ / float
#: types that appear in Unsloth "UD-*_K_XL" and stock GGUF checkpoints; an
#: unknown type falls back to 2 B/element (BF16-equivalent) with a warning.
_GGML_TYPE_SIZE: Dict[int, Tuple[int, int]] = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 36),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    30: (1, 2),     # BF16
}

#: GGUF metadata value-type enum (subset used by the header).
_GGUF_T_UINT8, _GGUF_T_INT8, _GGUF_T_UINT16, _GGUF_T_INT16 = 0, 1, 2, 3
_GGUF_T_UINT32, _GGUF_T_INT32, _GGUF_T_FLOAT32, _GGUF_T_BOOL = 4, 5, 6, 7
_GGUF_T_STRING, _GGUF_T_ARRAY, _GGUF_T_UINT64, _GGUF_T_INT64 = 8, 9, 10, 11
_GGUF_T_FLOAT64 = 12
_GGUF_SCALAR_FMT = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B",
    10: "Q", 11: "q", 12: "d",
}


#: Layer-type names that own a paged KV cache. Linear/mamba/GDN layers keep a
#: fixed-size recurrent STATE instead (sized by the mamba pool, not per token),
#: so they must not be counted here.
_KV_BEARING_LAYER_TYPES = ("full_attention", "sliding_attention", "attention")


def _kv_cell_bytes_from_config(cfg: dict, kv_cache_dtype: Optional[str]) -> Optional[float]:
    """KV bytes per token for ANY model config -- used to size an external
    speculative draft's KV pool from the DRAFT's own layout rather than
    assuming it matches the target's.

    Deliberately layout-generic, because a draft may be:

    * dense MHA/GQA -> 2 (K and V) * kv_heads * head_dim per layer;
    * MLA / DeepSeek-style -> ONE latent vector per token
      (kv_lora_rank + qk_rope_head_dim), not a K/V pair;
    * hybrid (GDN / mamba / linear attention mixed with attention) -> only the
      attention layers own a paged KV cache; the recurrent layers hold a
      fixed-size state that does not scale with context;
    * GGUF -> ``_load_config`` already synthesizes the same keys from the
      header, so this works unchanged.

    Returns None when the config does not describe a KV cache the caller can
    size, so callers keep their previous fallback rather than guessing.
    """
    if not isinstance(cfg, dict):
        return None
    text = cfg.get("text_config", cfg)
    elem = 1 if "fp8" in str(kv_cache_dtype or "") else 2

    # KV-bearing layer count: honour layer_types when present (hybrid models),
    # else assume every layer is an attention layer.
    layer_types = text.get("layer_types")
    if layer_types:
        layers = sum(1 for t in layer_types if str(t) in _KV_BEARING_LAYER_TYPES)
    else:
        layers = int(text.get("num_hidden_layers", 0) or 0)
    if layers <= 0:
        return None

    # MLA: a single compressed latent per token, so no factor 2 and no kv_heads.
    kv_lora_rank = int(text.get("kv_lora_rank", 0) or 0)
    if kv_lora_rank > 0:
        rope_dim = int(text.get("qk_rope_head_dim", 0) or 0)
        return float((kv_lora_rank + rope_dim) * elem * layers)

    # Dense MHA / GQA.
    kv_heads = int(
        text.get("num_key_value_heads")
        or text.get("num_attention_heads")
        or 0
    )
    head_dim = int(text.get("head_dim", 0) or 0)
    if head_dim <= 0:
        hidden = int(text.get("hidden_size", 0) or 0)
        q_heads = int(text.get("num_attention_heads", 0) or 0)
        head_dim = hidden // q_heads if hidden > 0 and q_heads > 0 else 0
    if kv_heads <= 0 or head_dim <= 0:
        return None
    return float(2 * kv_heads * head_dim * elem * layers)


def _is_gguf_model(model_path: Optional[str]) -> bool:
    """A GGUF checkpoint is addressed by a single .gguf file rather than a
    directory with config.json."""
    if not model_path:
        return False
    p = str(model_path)
    return p.lower().endswith(".gguf") or os.path.isfile(p)


def _load_checkpoint_config(model_path: str) -> dict:
    """The checkpoint's config dict, HF directory or GGUF file alike.

    GGUF checkpoints are a single .gguf FILE, not a directory with a
    config.json -- read the fields (and per-family quant bytes) from the GGUF
    header instead of crashing on ``open(model_path/'config.json')``."""
    if _is_gguf_model(model_path):
        return _gguf_config_and_families(model_path)
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


def _gguf_mmproj_bytes(model_path: Optional[str]) -> int:
    """On-disk bytes of the ``mmproj-*.gguf`` vision-encoder sidecar that ships
    beside a GGUF text checkpoint (0 when none). Added to the resident weight
    budget only when the plan includes the vision tower; text-only serving does
    not load it."""
    import glob as _glob

    if not model_path:
        return 0
    d = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    if not d:
        return 0
    best = 0
    for f in _glob.glob(os.path.join(d, "*.gguf")):
        b = os.path.basename(f).lower()
        if "mmproj" in b:
            best = max(best, os.path.getsize(f))
    return best


def _read_gguf_metadata(path: str) -> Tuple[Dict[str, object], Dict[str, int], List[dict]]:
    """Return (scalar KV dict, array-length dict, tensor-info list) from a
    GGUF file's header. Array values are NOT materialized (the token/merge
    arrays hold ~250k entries); only their length is recorded, which is all
    the cost model needs (vocab size). Tensor infos are
    [{name, dims, ggml_type}]."""
    import struct

    def rd(f, fmt: str):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    def rstr(f) -> str:
        n = rd(f, "Q")
        return f.read(n).decode("utf-8", "replace")

    def skip_array(f) -> int:
        et = rd(f, "I")
        n = rd(f, "Q")
        if et in _GGUF_SCALAR_FMT:
            f.seek(struct.calcsize(_GGUF_SCALAR_FMT[et]) * n, os.SEEK_CUR)
        elif et == _GGUF_T_STRING:
            for _ in range(n):
                f.seek(rd(f, "Q"), os.SEEK_CUR)
        elif et == _GGUF_T_ARRAY:
            for _ in range(n):
                skip_array(f)
        return n

    def read_value(f, t: int):
        if t in _GGUF_SCALAR_FMT:
            return rd(f, _GGUF_SCALAR_FMT[t])
        if t == _GGUF_T_STRING:
            return rstr(f)
        if t == _GGUF_T_ARRAY:
            return None  # length captured separately
        raise ValueError(f"unsupported GGUF value type {t}")

    scalars: Dict[str, object] = {}
    array_lens: Dict[str, int] = {}
    tensors: List[dict] = []
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        rd(f, "I")  # version
        n_tensors = rd(f, "Q")
        n_kv = rd(f, "Q")
        for _ in range(n_kv):
            key = rstr(f)
            t = rd(f, "I")
            if t == _GGUF_T_ARRAY:
                # Peek element type/count without consuming, then skip.
                array_lens[key] = skip_array(f)
            else:
                scalars[key] = read_value(f, t)
        for _ in range(n_tensors):
            name = rstr(f)
            nd = rd(f, "I")
            dims = [rd(f, "Q") for _ in range(nd)]
            gtype = rd(f, "I")
            rd(f, "Q")  # data offset (unused)
            tensors.append({"name": name, "dims": dims, "ggml_type": gtype})
    return scalars, array_lens, tensors


def _gguf_type_bytes_per_elem(gtype: int) -> float:
    entry = _GGML_TYPE_SIZE.get(gtype)
    if entry is None:
        logger.warning(
            "auto-performance: unknown ggml quant type %d in GGUF checkpoint; "
            "assuming 2 B/element for the family byte model.",
            gtype,
        )
        return 2.0
    block, tbytes = entry
    return tbytes / block


def _gguf_family_of(name: str) -> Optional[str]:
    """Map a GGUF tensor name onto the cost model's weight families. The GDN
    (linear-attention) in_proj is stored under ``attn_qkv``/``attn_gate`` in
    llama.cpp's qwen35 naming, so it is grouped with ``ssm_*`` into 'gdn';
    only the full-attention q/k/v/o projections are 'attn'."""
    if "nextn" in name or "mtp" in name:
        return "draft"
    # All FFN tensors -> 'mlp', INCLUDING the MoE expert stacks
    # (ffn_{gate,up,down}_exps) and the router (ffn_gate_inp). Excluding
    # ``exp`` here previously dropped the entire MoE expert mass -- the bulk
    # of a sparse-MoE GGUF -- from the family byte model.
    if "ffn_" in name:
        return "mlp"
    if name in ("token_embd.weight", "output.weight"):
        return "vocab"
    if "attn_qkv" in name or "attn_gate" in name or "ssm" in name or "conv" in name:
        return "gdn"
    if any(s in name for s in ("attn_q.", "attn_k.", "attn_v.", "attn_output")):
        return "attn"
    return None  # norms / biases (negligible mass) -> ignored in the avg


def _gguf_config_and_families(path: str) -> dict:
    """Synthesize an HF-config-shaped dict for the perf cost model from a
    GGUF header, plus measured per-family bytes/parameter and a few private
    hint keys (prefixed ``__gguf_``). Raises a clear error if a required
    field is missing instead of crashing later deep in the math."""
    scalars, array_lens, tensors = _read_gguf_metadata(path)
    arch = scalars.get("general.architecture")
    if not arch:
        raise ValueError(f"{path}: GGUF header lacks general.architecture")

    def need(key: str):
        full = f"{arch}.{key}"
        if full not in scalars:
            raise ValueError(
                f"auto-performance: GGUF checkpoint {path} is missing the "
                f"required metadata key '{full}'; cannot build the perf cost "
                f"model. Pin --rank-mlp-ratio to skip the optimizer."
            )
        return scalars[full]

    def opt(key: str, default):
        return scalars.get(f"{arch}.{key}", default)

    hidden = int(need("embedding_length"))
    # MoE GGUF (qwen35moe, ...) carries no dense ``feed_forward_length`` -- the
    # FFN mass is in ``expert_count`` experts of ``expert_feed_forward_length``
    # (plus an optional shared expert). Fall back to the expert width so the
    # cost model is well-defined; the MoE param count is rebuilt from these in
    # _build_families.
    expert_count = int(opt("expert_count", 0) or 0)
    expert_used = int(opt("expert_used_count", 0) or 0)  # top-k active experts
    expert_ffn = int(opt("expert_feed_forward_length", 0) or 0)
    shared_ffn = int(opt("expert_shared_feed_forward_length", 0) or 0)
    if expert_count > 0:
        intermediate = expert_ffn or int(opt("feed_forward_length", 0) or 0)
    else:
        intermediate = int(need("feed_forward_length"))
    q_heads = int(need("attention.head_count"))
    # Head geometry: prefer the ACTUAL attention projection tensors (ground
    # truth) over the header scalars, which some arches report inconsistently.
    # Gemma4 GGUFs omit head_count_kv entirely AND report a key_length (512)
    # that disagrees with the real attn_q width (4096 for head_count=16, i.e.
    # head_dim 256, not 512). Trusting the scalars would require a missing key
    # (hard error) and, if defaulted naively, 4x the KV-cache size + 2x the
    # q-proj mass. The weight tensors settle it unambiguously.
    def _blk0_dim(sub: str, index: int):
        for t in tensors:
            n = t["name"]
            if n.startswith("blk.0.") and f"{sub}." in n and n.endswith(".weight"):
                d = t.get("dims") or []
                return int(d[index]) if d else None  # GGML weight dims = [in, out]
        return None

    def _blk0_out(sub: str):
        return _blk0_dim(sub, -1)

    q_out = _blk0_out("attn_q")
    if q_out and q_heads and q_out % q_heads == 0:
        head_dim = q_out // q_heads
    else:
        head_dim = int(opt("attention.key_length", hidden // max(q_heads, 1)))
    kv_meta = scalars.get(f"{arch}.attention.head_count_kv")
    if kv_meta is not None:
        kv_heads = int(kv_meta)
    else:
        # head_count_kv absent (Gemma4 etc.): derive the GQA group count from
        # the real attn_k projection width; fall back to MHA (== q_heads) only
        # if the tensor is unavailable (fused QKV / unusual naming).
        k_out = _blk0_out("attn_k")
        kv_heads = (
            k_out // head_dim
            if (k_out and head_dim and k_out % head_dim == 0)
            else q_heads
        )
    # DeepSeek V4's wo_a projection couples heads and o_groups (#402, see
    # models/deepseek_v4.py MqaAttentionBase.__init__): its UNSHARDED
    # per-group input width is `n_heads * head_dim // o_groups`, which is
    # exactly the GGML "in" dim (dims[0]) of the on-disk attn_output_a
    # tensor -- ColumnParallelLinear(input_size=n_heads*head_dim//o_groups,
    # output_size=o_groups*o_lora_rank). Unlike head_dim/kv_heads above,
    # o_groups has NO equivalent GGUF header scalar at all (it is a
    # fork-specific unit llama.cpp's deepseek4 writer never declares), so
    # the tensor is the ONLY source -- and it is an exact closed form, not a
    # heuristic: any DSV4-family GGUF with this tensor yields the true
    # o_groups, and every non-DSV4 arch simply lacks the tensor, so
    # `o_groups` stays 0 and `PerfCostModel.attn_units` keeps gridding on
    # kv_heads exactly as before (#414).
    wo_a_in = _blk0_dim("attn_output_a", 0)
    o_groups = (
        (q_heads * head_dim) // wo_a_in
        if wo_a_in and q_heads and head_dim and (q_heads * head_dim) % wo_a_in == 0
        else 0
    )
    block_count = int(need("block_count"))
    nextn = int(opt("nextn_predict_layers", 0) or 0)
    n_layers = block_count - nextn  # nextn/MTP block is not a base layer
    interval = int(opt("full_attention_interval", 0) or 0)
    if interval > 0:
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(n_layers)
        ]
    else:
        layer_types = None  # __init__ falls back to all-full

    # Vocab size = length of the token list (arrays are not materialized).
    vocab = (
        array_lens.get("tokenizer.ggml.tokens")
        or array_lens.get("tokenizer.ggml.token_type")
        or int(opt("vocab_size", 0) or 0)
    )

    # GDN / SSM geometry (llama.cpp ssm.* keys).
    gdn_k_heads = int(opt("ssm.group_count", 0) or 0)
    gdn_v_heads = int(opt("ssm.time_step_rank", 0) or 0)
    gdn_k_dim = int(opt("ssm.state_size", 0) or 0)
    ssm_inner = int(opt("ssm.inner_size", 0) or 0)
    gdn_v_dim = (ssm_inner // gdn_v_heads) if gdn_v_heads else gdn_k_dim
    conv_kernel = int(opt("ssm.conv_kernel", 4) or 4)

    # Per-family bytes/element from the tensor quant types (element-weighted).
    fam_bytes: Dict[str, float] = {}
    fam_elems: Dict[str, float] = {}
    attn_q_out = 0
    has_draft_body = False
    for t in tensors:
        fam = _gguf_family_of(t["name"])
        dims = t["dims"]
        elems = 1
        for d in dims:
            elems *= d
        if "attn_q." in t["name"] and len(dims) >= 2:
            attn_q_out = max(attn_q_out, max(dims))
        if "nextn" in t["name"] and any(
            s in t["name"] for s in ("attn_q", "attn_k", "attn_v", "ffn_")
        ):
            has_draft_body = True
        if fam is None:
            continue
        bpe = _gguf_type_bytes_per_elem(t["ggml_type"])
        fam_bytes[fam] = fam_bytes.get(fam, 0.0) + elems * bpe
        fam_elems[fam] = fam_elems.get(fam, 0.0) + elems
    family_bpp = {
        fam: (fam_bytes[fam] / fam_elems[fam]) for fam in fam_bytes if fam_elems[fam]
    }
    # Reasonable fallbacks if a family had no matched tensors.
    family_bpp.setdefault("attn", 1.0)
    family_bpp.setdefault("mlp", 0.75)
    family_bpp.setdefault("gdn", 2.0)
    family_bpp.setdefault("vocab", 1.0625)
    family_bpp.setdefault("draft", 1.0625)

    # attn_output_gate: the full-attention q projection emits q + output gate
    # (2x q_heads*head_dim) when gating is on. Detect from the tensor shape;
    # default False if no separate attn_q tensor was found.
    attn_gate = bool(attn_q_out >= 2 * q_heads * head_dim and attn_q_out > 0)

    # MLP shard granularity: the down-proj is quantized in blocks along the
    # contracted (intermediate) axis, so the natural indivisible unit is the
    # dominant MLP quant block (K-quants = 256, Q8_0 = 32).
    mlp_types = [t["ggml_type"] for t in tensors if _gguf_family_of(t["name"]) == "mlp"]
    if mlp_types:
        dominant = max(set(mlp_types), key=mlp_types.count)
        mlp_group = _GGML_TYPE_SIZE.get(dominant, (256, 0))[0]
    else:
        mlp_group = 256

    return {
        "text_config": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": n_layers,
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "o_groups": o_groups,
            "attn_output_gate": attn_gate,
            "vocab_size": int(vocab),
            "num_experts": expert_count,
            "num_experts_per_tok": expert_used if expert_count else 0,
            "moe_intermediate_size": expert_ffn if expert_count else 0,
            "shared_expert_intermediate_size": shared_ffn if expert_count else 0,
            "mtp_num_hidden_layers": nextn,
            "linear_num_key_heads": gdn_k_heads,
            "linear_num_value_heads": gdn_v_heads,
            "linear_key_head_dim": gdn_k_dim,
            "linear_value_head_dim": gdn_v_dim,
            "linear_conv_kernel_dim": conv_kernel,
            "layer_types": layer_types,
        },
        "quantization_config": {"group_size": mlp_group},
        "__gguf_family_bpp__": family_bpp,
        "__gguf_has_draft_body__": has_draft_body,
    }


#: The weight families a linear-layer quantization scheme actually quantizes
#: (attention q/k/v/o, dense-or-MoE MLP, and the MTP/draft body). Embeddings/
#: lm_head (``vocab``), the SSM/GDN state (``gdn``), the vision tower
#: (``vision``) and the draft fc (``draft_repl``) stay at their native dtype
#: in every AWQ/GPTQ/FP8/compressed-tensors checkpoint we size, exactly as the
#: on-disk anchoring path treats them -- so the two paths agree.
_QUANTIZABLE_FAMILIES = ("attn", "mlp", "draft_attn", "draft_mlp")

#: Representative tensor path per quantizable family, matched against a
#: scheme's per-module exclusion patterns (gptq ``dynamic`` ``-:<regex>``,
#: compressed-tensors ``ignore``, ``modules_to_not_convert``) to decide
#: whether the family is quantized or kept at its native dtype.
_FAMILY_REP_NAMES = {
    "attn": "model.language_model.layers.0.self_attn.q_proj.weight",
    "mlp_dense": "model.language_model.layers.0.mlp.gate_proj.weight",
    "mlp_moe": "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
    "gdn": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "draft_attn": "model.model.mtp.layers.0.self_attn.q_proj.weight",
    "draft_mlp": "model.model.mtp.layers.0.mlp.gate_proj.weight",
}



def _int_quant_bpp(bits: float, group_size: Optional[int], symmetric: bool) -> float:
    """Bytes/param for a grouped integer quant (AWQ/GPTQ/compressed-tensors):
    ``bits/8`` packed weight + an fp16 scale per group per output channel
    (``2/group``) + an int4 zero-point per group when asymmetric
    (``0.5/group``). Ungrouped (per-channel) scales are negligible per param."""
    bpp = bits / 8.0
    if group_size and group_size > 0:
        bpp += 2.0 / group_size  # fp16 group scale
        if not symmetric:
            bpp += 0.5 / group_size  # packed int4 zero-point
    return bpp


def _family_broadly_excluded(rep_name: str, regex_pats: List[str]) -> bool:
    """True when a WHOLE weight family is kept at native dtype by a scheme's
    broad, family-level exclusion -- a ``dynamic`` ``-:<regex>`` (gptq) or a
    compressed-tensors ``re:<regex>``. These are the only authoritative
    family-wide signals; the Qwen MoE GPTQ configs use them to keep all of
    attention / mtp / shared_expert in higher precision.

    Fine-grained LITERAL lists (``modules_to_not_convert`` / ``ignore``) are
    deliberately NOT consulted here: in these checkpoints they mix precision
    WITHIN a family and across a SUBSET of layers (routed experts INT4 but the
    shared-expert + router + a per-layer selection of self_attn/linear_attn
    kept BF16). Mapping such a partial, intra-family list onto a single
    per-family bytes/param would mis-size the dominant mass far worse than
    treating the family at its nominal scheme width; the routed/dense bulk is
    what the scheme width describes, and the BF16 remainder is a few percent
    that stays within the sizing tolerance. The measured on-disk anchor (used
    whenever the weight shards are present) captures the exact mix regardless."""
    for pat in regex_pats:
        try:
            if re.search(pat, rep_name):
                return True
        except re.error:
            if pat and pat in rep_name:
                return True
    return False


def _config_quant_bpp(cfg: dict, is_moe: bool) -> Optional[Dict[str, float]]:
    """Config-AUTHORITATIVE bytes/param for the quantized weight families,
    read from the checkpoint's own ``quantization_config`` -- never inferred
    from the repo/path name.

    Returns ``{family_name: bytes_per_param}`` for the families the scheme
    quantizes, or ``None`` when the config declares no quantization (a plain
    bf16/fp16 checkpoint, where every family stays 2 B/param).

    This is what lets a config-only HF-hub snapshot (a hash-named dir with the
    weight shards absent) size IDENTICALLY to the same repo given as a local
    directory: the byte model no longer silently falls back to BF16 when the
    .safetensors files are not on disk. When the shards ARE present the
    measured checkpoint size still wins in ``_build_families`` (it captures
    every ``modules_to_not_convert`` exception exactly); this is the
    no-weights path.

    Honors, generically:
      * ``quant_method``: fp8 (1 B/weight + a negligible block/channel scale),
        awq / gptq / compressed-tensors int (``bits/8`` + fp16 group scales +
        int zero-points when asymmetric);
      * broad family-level exclusions -- gptq ``dynamic`` ``-:<regex>`` and
        compressed-tensors ``re:<regex>`` -- so a family the scheme keeps in
        higher precision (attn / mtp in the Qwen MoE GPTQ configs) stays at
        2 B/param. MIXED-precision checkpoints that keep only a SUBSET
        (shared-expert, router, or a per-layer selection of GDN/attention) at
        BF16 via literal ``modules_to_not_convert`` / ``ignore`` lists are sized
        at the routed/dense scheme width (the mass), with the small BF16
        remainder inside the sizing tolerance; the on-disk anchor sizes the
        exact mix whenever the weight shards are present.
    """
    qc = _quant_config(cfg)
    if not qc:
        return None

    method = str(qc.get("quant_method") or "").lower()
    fmt = str(qc.get("format") or qc.get("fmt") or "").lower()

    # Broad family-level exclusion regexes (see _family_broadly_excluded).
    regex_pats: List[str] = []
    for entry in (qc.get("ignore") or []):
        s = str(entry)
        if s.startswith("re:"):
            regex_pats.append(s[3:])
    dynamic = qc.get("dynamic") or {}
    for key in dynamic:
        if str(key).startswith("-:"):
            regex_pats.append(str(key)[2:])

    # Bytes/param for the quantized (non-excluded) families. Same predicate
    # the lane dispatch uses, so a checkpoint cannot be SIZED as fp8 and
    # SCORED as something else.
    fp8_like = _is_fp8_like(method, fmt)
    if fp8_like:
        # e4m3/e5m2 weights are 1 B; block (weight_block_size) or per-channel
        # fp32 scales add a negligible per-param overhead.
        quant_bpp = 1.0
    else:
        # Grouped integer schemes: awq / gptq / compressed-tensors.
        bits = qc.get("bits")
        symmetric = qc.get("sym")
        group = qc.get("group_size")
        groups = qc.get("config_groups") or {}
        if groups:
            w = (next(iter(groups.values())) or {}).get("weights") or {}
            bits = bits if bits is not None else w.get("num_bits")
            group = group if group is not None else w.get("group_size")
            if symmetric is None:
                symmetric = w.get("symmetric")
        if bits is None:
            bits = 4.0 if method in ("awq", "gptq") else None
        if bits is None:
            # Unknown scheme with no bit width -> cannot size authoritatively.
            return None
        # AutoGPTQ/GPTQModel + AWQ always materialize a packed zero-point tensor
        # (``qzeros``) on disk regardless of the ``sym`` flag; only
        # compressed-tensors genuinely drops zeros when symmetric. So the
        # zero-point storage term is present unless it is symmetric
        # compressed-tensors.
        symmetric_storage = bool(symmetric) and method not in ("awq", "gptq")
        quant_bpp = _int_quant_bpp(float(bits), group, symmetric_storage)

    # FP8 block/channel quantization targets EVERY Linear, so the GDN/linear-
    # attention in_proj + out_proj are fp8 too (only the vision tower is in
    # modules_to_not_convert). Grouped-integer AWQ/GPTQ, by contrast, leaves
    # the SSM/GDN mixer at its native dtype -- so ``gdn`` is a quantized family
    # only under fp8-like schemes. (The gdn family also folds in small bf16
    # conv/norm/A/dt tensors; those are a few % of its mass, well within the
    # sizing tolerance.)
    families = list(_QUANTIZABLE_FAMILIES)
    if fp8_like:
        families.append("gdn")

    rep_for = {
        "mlp": _FAMILY_REP_NAMES["mlp_moe" if is_moe else "mlp_dense"],
    }
    out: Dict[str, float] = {}
    for fam in families:
        rep = rep_for.get(fam, _FAMILY_REP_NAMES.get(fam, fam))
        if _family_broadly_excluded(rep, regex_pats):
            continue  # kept at native dtype -> caller leaves it at 2 B/param
        out[fam] = quant_bpp
    return out or None


class PerfCostModel:
    """Parse-time capacity/speed predictor for MLP-vector candidates.

    All quantities are derived from config.json + the on-disk checkpoint
    size + the resolved --rank-tp-ratio auto budgets, BEFORE any weights are
    loaded. Absolute token numbers are estimates (logged as such); the floor
    decision only consumes candidate-over-base RATIOS, whose dominant term
    (MLP bytes per unit) is exact.
    """

    #: Class-level default so the bandwidth/prefill methods stay callable on
    #: a bare instance (tests build one via ``__new__`` to exercise them
    #: config-free). All-None fields read the module constants at call time;
    #: ``__init__`` replaces this with the env-aware resolution.
    calibration: PerfCalibration = PerfCalibration()

    def __init__(
        self,
        plan_inputs,
        base_plan: List[int],
        budgets_mib: List[int],
        measured: Optional[List[dict]] = None,
        measured_mlp_vector: Optional[List[int]] = None,
        calibration: Optional[PerfCalibration] = None,
    ):
        # ``plan_inputs`` is a PlanInputs dataclass (see above). The boot
        # path builds it via PlanInputs.from_server_args so the server and
        # the offline planner (sglang.srt.planner) run identical sizing.
        #
        # ``measured``: optional per-rank residency posts from the measured
        # KV-budget registry (load_measured_components). When present,
        # predict_capacity switches from the static-heuristic budget model to
        # the measured one (see there); ``measured_mlp_vector`` is the weight
        # vector the measurement was taken under, used to anchor the family
        # model's absolute weight bytes against the measured allocator value.
        # ``calibration``: the fitted/assumed scalars of the speed model (the
        # refit seam, see PerfCalibration). Default: env overrides over the
        # shipped reference-rig fit.
        self.tp_size = plan_inputs.tp_size
        self.base_plan = list(base_plan)
        self.budgets_mib = list(budgets_mib)
        self.plan_inputs = plan_inputs
        self.calibration = (
            calibration if calibration is not None else PerfCalibration.from_env()
        )
        self.measured = (
            list(measured)
            if measured is not None and len(measured) == self.tp_size
            else None
        )
        self.measured_mlp_vector = (
            [int(v) for v in measured_mlp_vector]
            if measured_mlp_vector is not None
            and len(measured_mlp_vector) == self.tp_size
            else None
        )

        cfg = self._load_config(plan_inputs.model_path)
        text = cfg.get("text_config", cfg)
        self.hidden = int(text["hidden_size"])
        # MoE geometry: a sparse-MoE checkpoint (Qwen3.5-MoE, DeepSeek, ...)
        # carries no dense ``intermediate_size`` -- the FFN mass lives in
        # ``num_experts`` routed experts of width ``moe_intermediate_size``
        # (plus an optional shared expert). Reading ``intermediate_size``
        # unconditionally KeyError'd on these; and even when a dense
        # intermediate existed, ignoring the experts undercounted the weights
        # by ~10x. Fall back to the MoE width so the unit grid + sizing below
        # are well-defined for both dense and MoE checkpoints.
        self.num_experts = int(
            text.get("num_experts", text.get("n_routed_experts", 0)) or 0
        )
        # Active (routed) experts per token — the MoE sparsity factor the
        # roofline estimate needs to size "active bytes/params per token"
        # (decode streams only the top-k experts, not all of them). HF configs
        # name it ``num_experts_per_tok`` (some ``num_activated_experts``); the
        # GGUF synthesizer maps ``expert_used_count`` to the same key. Falls
        # back to 8 (the common Qwen/Mixtral default) if a MoE checkpoint omits
        # it, and is clamped to [1, num_experts].
        self.num_experts_per_tok = int(
            text.get("num_experts_per_tok")
            or text.get("num_activated_experts")
            or (8 if self.num_experts > 0 else 0)
        )
        if self.num_experts > 0:
            self.num_experts_per_tok = max(
                1, min(self.num_experts_per_tok, self.num_experts)
            )
        self.moe_intermediate = int(text.get("moe_intermediate_size", 0) or 0)
        self.shared_expert_intermediate = int(
            text.get("shared_expert_intermediate_size", 0) or 0
        )
        self.intermediate = int(
            text.get("intermediate_size")
            or self.moe_intermediate
            or 0
        )
        layer_types = text.get("layer_types")
        n_layers = int(text["num_hidden_layers"])
        if layer_types:
            self.full_layers = sum(1 for t in layer_types if t == "full_attention")
            self.gdn_layers = sum(1 for t in layer_types if t == "linear_attention")
        else:
            self.full_layers, self.gdn_layers = n_layers, 0
        self.n_layers = n_layers
        # #371: per-layer family census. `layer_types` covers the hybrid
        # (Qwen-style) shape; a HETEROGENEOUS-LAYER checkpoint declares its
        # per-layer blocks in `block_configs` instead (Nemotron-NAS / Puzzle:
        # each entry carries attention.no_op / ffn.no_op and its own
        # ffn_mult), and that shape reaches the `else` branch above, where
        # every layer is counted as a full-attention MLP layer. The census
        # corrects both family counts and returns the FFN width factor, so a
        # variable-width FFN stack is summed rather than multiplied.
        self.layer_census = layer_family_census(text, n_layers)
        if self.layer_census.heterogeneous:
            self.full_layers = self.layer_census.attn_layers
        self.kv_heads = int(text.get("num_key_value_heads", 1))
        self.q_heads = int(text.get("num_attention_heads", 1))
        self.head_dim = int(text.get("head_dim", self.hidden // self.q_heads))
        self.attn_gate = bool(text.get("attn_output_gate", False))
        self.vocab = int(text.get("vocab_size", 0))
        self.gdn_k_heads = int(text.get("linear_num_key_heads", 0) or 0)
        self.gdn_v_heads = int(text.get("linear_num_value_heads", 0) or 0)
        self.gdn_k_dim = int(text.get("linear_key_head_dim", 0) or 0)
        self.gdn_v_dim = int(text.get("linear_value_head_dim", 0) or 0)
        self.conv_kernel = int(text.get("linear_conv_kernel_dim", 4) or 4)
        self.mtp_layers = int(text.get("mtp_num_hidden_layers", 0) or 0)

        # Unit grids (must match the model's tp_units so candidate vectors
        # materialize identically to the real partition).
        #
        # DeepSeek V4 declares ``o_groups`` and pins num_key_value_heads to 1:
        # its attention block partitions in whole o_groups (heads and o_groups
        # are coupled through wo_a, see models/deepseek_v4.py), so the kv-head
        # count describes nothing there. Any config without ``o_groups`` keeps
        # the kv-head grid unchanged.
        self.attn_units = int(text.get("o_groups") or 0) or max(self.kv_heads, 1)
        self.gdn_units = max(self.gdn_k_heads, 1)
        quant_cfg = cfg.get("quantization_config") or {}
        group = self._quant_group_size(quant_cfg)
        if self.num_experts > 0:
            # MoE shards whole experts across ranks (the fork's --rank-moe-ratio
            # granularity), so the natural indivisible unit is one expert, not
            # a quant-group column of the (tiny) expert intermediate.
            self.mlp_units = self.num_experts
        elif group and self.intermediate % group == 0:
            self.mlp_units = self.intermediate // group
        elif self.intermediate % 128 == 0:
            self.mlp_units = self.intermediate // 128
        else:
            self.mlp_units = math.gcd(self.intermediate, 512) or 1

        self.spec_active = plan_inputs.speculative_algorithm is not None
        self.spec_draft_tokens = int(plan_inputs.speculative_num_draft_tokens or 0)
        # -- draft PLACEMENT (--speculative-draft-placement) -----------------
        # Split (default): every rank carries ~1/tp of the draft, which is what
        # the draft_* families below encode. Solo: ONE rank carries the whole
        # unsharded draft and the others carry none -- the exact opposite
        # profile. Charging the split cost under solo hands the solo host
        # (usually the fastest card, so the one the optimizer already wants to
        # load) a weight shard it has no room for, collapsing its KV capacity
        # and -- because the global context is min_r(P_r/ratio_r)*sum(ratios)
        # -- throttling every other rank too. Mirrors the token planner's
        # pool_configurator.solo_draft_kv_cell_factor.
        # Cross-algorithm gate (T156): the GLOBAL placement stays "split"
        # (NEXTN/MTP rung), but the DFLASH rung's draft — its whole external
        # checkpoint plus a draft KV pool sized to the GLOBAL context — is
        # always solo-resident on rank 0. For the capacity math that is the
        # solo profile (solo ckpt bytes + global-context draft cell on the
        # host), so solo_active covers it; the draft_* family re-pointing
        # below must NOT happen though, because those families model the
        # target-derived MTP/NEXTN draft, which stays split under cross.
        self.cross_active = bool(
            self.spec_active
            and getattr(plan_inputs, "speculative_cross_algorithm", False)
            and self.tp_size >= 2
        )
        self._placement_solo = bool(
            self.spec_active
            and str(getattr(plan_inputs, "speculative_draft_placement", "split"))
            == "solo"
            and self.tp_size >= 2
        )
        self.solo_active = self._placement_solo or self.cross_active
        self.solo_rank = (
            int(getattr(plan_inputs, "speculative_draft_solo_rank", 0) or 0)
            if self._placement_solo
            else 0
        )
        if not (0 <= self.solo_rank < self.tp_size):
            self.solo_rank = 0
        # External draft checkpoint (DFLASH / any --speculative-draft-model-path):
        # its bytes live in a SEPARATE checkpoint, so the target-config-derived
        # draft_attn/draft_mlp mass above does not describe it at all. Only
        # counted under solo, where it is unambiguously one rank's resident
        # cost; the split path keeps its historical (unmodelled) behaviour so
        # non-solo planning stays byte-identical.
        self.solo_draft_ckpt_bytes = 0.0
        #: Draft-KV bytes per token on the solo host. ``None`` = fall back to
        #: the target's mtp_layers-derived term.
        self.solo_draft_kv_cell_bytes = None
        if self.solo_active and plan_inputs.speculative_draft_model_path:
            from sglang.srt.distributed.utils import _checkpoint_size_mib

            self.solo_draft_ckpt_bytes = float(
                _checkpoint_size_mib(plan_inputs.speculative_draft_model_path)
            ) * 2**20
            # An EXTERNAL draft (DFLASH) has its own depth and KV geometry, so
            # the target's ``mtp_num_hidden_layers`` describes a different model
            # entirely. Concretely on the reference rig the DFLASH draft is
            # 5 layers x 8 kv heads while the target's MTP term is 1 layer x
            # 4 kv heads -- a 10x under-count of a pool that is sized to the
            # GLOBAL context, i.e. multiple GB on the host. Read the draft's
            # own config instead; fall back silently if it is unreadable.
            try:
                self.solo_draft_kv_cell_bytes = _kv_cell_bytes_from_config(
                    self._load_config(plan_inputs.speculative_draft_model_path),
                    plan_inputs.kv_cache_dtype,
                )
            except Exception:  # pragma: no cover - defensive, config optional
                self.solo_draft_kv_cell_bytes = None
        kv_dtype = str(plan_inputs.kv_cache_dtype or "auto")
        self.kv_cell_bytes_per_layer = (
            2 * self.kv_heads * self.head_dim * (1 if "fp8" in kv_dtype else 2)
        )
        cell_layers = self.full_layers + (self.mtp_layers if self.spec_active else 0)
        #: Full-kv-head KV bytes per token (weighted DCP replicates heads).
        self.kv_cell_bytes = self.kv_cell_bytes_per_layer * cell_layers

        self.families = self._build_families(cfg)
        self.mamba_pool_bytes = self._mamba_pool_bytes()

        # -- measured-registry refinements (all None/0 without a registry) ---
        #: Per-rank additive correction anchoring the family model's ABSOLUTE
        #: weight bytes to the measured post-weights allocator value at the
        #: vector the measurement was taken under. The family model's DELTAS
        #: between vectors stay model-derived (exact for the dominant MLP
        #: term); the bias removes its absolute error (buffers, fused/aux
        #: tensors, quant scale layouts the config-level model cannot see).
        self.measured_weight_bias = [0.0] * self.tp_size
        if self.measured is not None and self.measured_mlp_vector is not None:
            model_w = self.per_rank_weight_bytes(self.measured_mlp_vector)
            for r in range(self.tp_size):
                w_meas = float(
                    self.measured[r].get("weights_alloc_bytes", 0) or 0
                )
                if w_meas > 0:
                    self.measured_weight_bias[r] = w_meas - model_w[r]
        # Measured solo-draft KV cell (bytes per GLOBAL token): the host's
        # DFLASH pool size divided by the global token count it was sized to.
        # Preferred over the config-derived cell — it reflects the pool's
        # real page/layout overheads.
        if self.measured is not None and self.solo_active:
            host = self.measured[self.solo_rank]
            pool_b = float(host.get("draft_solo_pool_bytes", 0) or 0)
            tokens = float(host.get("max_total_num_tokens", 0) or 0)
            if pool_b > 0 and tokens > 0:
                self.solo_draft_kv_cell_bytes = pool_b / tokens
        # Measured TARGET-KV cell (bytes per pool token): pool bytes over
        # pool token slots, identical across ranks (each holds its token
        # share at the same cell). Removes the config-model's layer-count
        # bias (measured 2026-07-22: model 34816 vs real 32768 B/token on
        # the reference rig, a 6% capacity skew). Measured mode only — the
        # heuristic fallback keeps the config-derived cell byte-identically.
        if self.measured is not None:
            for c in self.measured:
                b = float(c.get("kv_pool_bytes", 0) or 0)
                t = float(c.get("kv_pool_tokens", 0) or 0)
                if b > 0 and t > 0:
                    self.kv_cell_bytes = b / t
                    break

    _load_config = staticmethod(_load_checkpoint_config)

    @staticmethod
    def _quant_group_size(quant_cfg: dict) -> Optional[int]:
        groups = quant_cfg.get("config_groups") or {}
        for g in groups.values():
            w = (g or {}).get("weights") or {}
            gs = w.get("group_size")
            if gs:
                return int(gs)
        gs = quant_cfg.get("group_size")
        return int(gs) if gs else None

    def _mlp_layer_factor(self) -> float:
        """Layer-equivalents of MLP mass in this checkpoint (#371).

        ``n_layers`` for every uniform stack -- so the family table is
        byte-identical for everything this fork serves today -- and the
        census's summed width factor for a heterogeneous one.
        """
        census = getattr(self, "layer_census", None)
        if census is None or census.uniform:
            return float(self.n_layers)
        return census.ffn_width_factor

    def _build_families(self, cfg: dict) -> Dict[str, _Family]:
        H, I = self.hidden, self.intermediate
        q_size = self.q_heads * self.head_dim * (2 if self.attn_gate else 1)
        kv_size = self.kv_heads * self.head_dim
        attn_layer = H * (q_size + 2 * kv_size) + (self.q_heads * self.head_dim) * H
        if self.num_experts > 0:
            # MoE FFN mass: num_experts routed experts (gate+up+down = 3 H*Wi)
            # of width moe_intermediate, plus an optional always-on shared
            # expert. The router gate (H*num_experts) is negligible. This is
            # the bulk of a sparse-MoE checkpoint -- omitting it undercounted
            # weights ~10x and produced spurious "fits" / mis-sized budgets.
            moe_i = self.moe_intermediate or I
            mlp_layer = self.num_experts * 3 * H * moe_i
            if self.shared_expert_intermediate > 0:
                mlp_layer += 3 * H * self.shared_expert_intermediate
        else:
            mlp_layer = 3 * H * I

        gdn_layer = 0.0
        if self.gdn_layers:
            k_sz = self.gdn_k_heads * self.gdn_k_dim
            v_sz = self.gdn_v_heads * self.gdn_v_dim
            # in_proj (q,k,v,z) + b/a + out_proj + conv + norms (approx).
            gdn_layer = (
                H * (2 * k_sz + 2 * v_sz)
                + H * 2 * self.gdn_v_heads
                + v_sz * H
                + (2 * k_sz + v_sz) * self.conv_kernel
            )

        vocab_params = 2.0 * self.vocab * H  # embed + lm_head (untied worst case)

        # Vision tower (VL checkpoints). Counted in the resident weight budget
        # only when the rig serves the model WITH vision. Text-only serving does
        # not load the encoder, so those bytes free up for KV cache -> more
        # tokens. The vision-off toggle flows in via ``include_vision``.
        # ``vision_disk_bytes`` is the tower's on-disk footprint -- subtracted
        # from the checkpoint anchor below when vision is off (an HF VL
        # checkpoint bundles the encoder into its .safetensors, so the anchor
        # must shed those bytes, not merely redistribute them).
        include_vision = getattr(self.plan_inputs, "include_vision", True)
        vision_params = 0.0
        vision_bpp = 2.0
        vision_disk_bytes = 0.0
        vcfg = cfg.get("vision_config")
        if vcfg and not cfg.get("language_model_only", False):
            vh = int(vcfg.get("hidden_size", 0) or 0)
            vi = int(vcfg.get("intermediate_size", 0) or 0)
            vd = int(vcfg.get("depth", 0) or 0)
            full_vision = vd * (4 * vh * vh + 2 * vh * vi)
            vision_disk_bytes = full_vision * 2.0  # unquantized (bf16) encoder
            if include_vision:
                vision_params = full_vision
        elif cfg.get("__gguf_family_bpp__") is not None:
            # GGUF: the vision encoder is a separate ``mmproj-*.gguf`` sidecar
            # beside the text checkpoint (NOT part of the sized .gguf), so it is
            # additive when on and simply omitted when off.
            mmproj_bytes = _gguf_mmproj_bytes(self.plan_inputs.model_path)
            if include_vision and mmproj_bytes > 0:
                vision_params = float(mmproj_bytes)
                vision_bpp = 1.0

        draft_attn = draft_mlp = draft_repl = 0.0
        if self.spec_active and self.mtp_layers:
            draft_attn = self.mtp_layers * attn_layer
            draft_mlp = self.mtp_layers * mlp_layer
            draft_repl = 2 * H * H  # fc (2H -> H), bf16, replicated
            # NOTE eb764a12b: the draft's embed/lm_head duplicates are shared
            # with the target BEFORE KV profiling, so they are a load-time
            # transient only and deliberately NOT part of this budget model.

        families = {
            # #371: MLP mass is the census's FFN WIDTH FACTOR, not the layer
            # count -- identical (== n_layers) for every uniform stack, and
            # for a heterogeneous one it both drops the no_op-FFN layers and
            # sums the per-layer widths instead of multiplying one width by
            # every layer. `full_layers` already carries the attention
            # correction (set from the census at parse time).
            "attn": _Family(self.full_layers * attn_layer, 2.0, "attn"),
            "gdn": _Family(self.gdn_layers * gdn_layer, 2.0, "gdn"),
            "mlp": _Family(
                self._mlp_layer_factor() * mlp_layer, 2.0, "mlp"
            ),
            "vocab": _Family(vocab_params, 2.0, "even"),
            "vision": _Family(vision_params, vision_bpp, "gdn_base"),
            "draft_attn": _Family(draft_attn, 2.0, "attn"),
            "draft_mlp": _Family(draft_mlp, 2.0, "mlp"),
            "draft_repl": _Family(draft_repl, 2.0, "replicated"),
        }

        # Solo placement: the draft is not sharded, it is RESIDENT ON ONE RANK.
        # Re-point every draft family at that rank (shadows drop to zero) and
        # add the external draft checkpoint, which the target-config-derived
        # mass above cannot describe. Guarded on solo_active so the split cost
        # model -- families, shards and bytes_per_param anchoring alike -- is
        # untouched.
        if self.solo_active:
            # Placement-solo: the target-derived draft families (MTP/NEXTN
            # mass) move to the host too. Cross gate: they STAY split (the
            # NEXTN rung shards as usual); only the external DFLASH
            # checkpoint below is host-resident.
            if self._placement_solo:
                for _name in ("draft_attn", "draft_mlp", "draft_repl"):
                    families[_name].shard = "solo_host"
            if self.solo_draft_ckpt_bytes > 0:
                families["draft_solo_ckpt"] = _Family(
                    self.solo_draft_ckpt_bytes, 1.0, "solo_host"
                )

        # GGUF path: every family (embed/lm_head, attention, MLP, GDN) is
        # quantized on its own ggml grid, so instead of the safetensors
        # "BF16-inside-INT4" anchoring below we set each family's bytes/param
        # directly from the measured per-family quant types (element-weighted
        # bytes/element read from the tensor-info block). Assumptions: norms/
        # biases are folded into their family's average (negligible mass); the
        # GDN in_proj (llama.cpp ``attn_qkv``) is counted in 'gdn'; when the
        # nextn/MTP block carries no own attn/ffn weights (module sharing) its
        # draft_attn/draft_mlp mass is zeroed so it is not double-counted.
        gguf_bpp = cfg.get("__gguf_family_bpp__")
        if gguf_bpp is not None:
            families["attn"].bytes_per_param = gguf_bpp["attn"]
            families["mlp"].bytes_per_param = gguf_bpp["mlp"]
            families["gdn"].bytes_per_param = gguf_bpp["gdn"]
            families["vocab"].bytes_per_param = gguf_bpp["vocab"]
            families["draft_repl"].bytes_per_param = gguf_bpp.get("draft", 2.0)
            families["draft_attn"].bytes_per_param = gguf_bpp["attn"]
            families["draft_mlp"].bytes_per_param = gguf_bpp["mlp"]
            if not cfg.get("__gguf_has_draft_body__", False):
                families["draft_attn"].params = 0.0
                families["draft_mlp"].params = 0.0
            return families

        # Anchor quantized-family bytes/param on the measured checkpoint
        # size: BF16 families (GDN, embed/lm_head, vision, draft fc -- the
        # "BF16 inside INT4" cost-model term) stay at 2 B/param, the
        # remaining checkpoint bytes are spread over the quantized families
        # (attn + MLP + draft layer) proportionally to their param counts.
        from sglang.srt.distributed.utils import _checkpoint_size_mib

        ckpt_bytes = _checkpoint_size_mib(self.plan_inputs.model_path) * 2**20
        # Text-only serving: shed the vision tower's bytes from the anchor (an
        # HF VL checkpoint bundles the unquantized encoder into its shards, so
        # its bytes must LEAVE the total, not be redistributed onto attn/MLP).
        if not include_vision and vision_disk_bytes > 0 and ckpt_bytes > 0:
            ckpt_bytes = max(ckpt_bytes - vision_disk_bytes, 0.0)
        quant_names = ("attn", "mlp", "draft_attn", "draft_mlp")
        # ``draft_solo_ckpt`` holds bytes from a SEPARATE checkpoint, so it must
        # not enter the anchoring that spreads THIS checkpoint's remaining bytes
        # over the quantized families -- counting it would deflate their
        # bytes/param. (Absent unless solo_active, so the split path is
        # unaffected.)
        bf16_bytes = sum(
            fam.bytes
            for name, fam in families.items()
            if name not in quant_names and name != "draft_solo_ckpt"
        )
        quant_params = sum(families[name].params for name in quant_names)
        if ckpt_bytes > 0 and quant_params > 0:
            bpp = (ckpt_bytes - bf16_bytes) / quant_params
            bpp = min(max(bpp, 0.5), 2.25)  # int4+scales ... bf16 bounds
            for name in quant_names:
                families[name].bytes_per_param = bpp
            return families

        # No weight files on disk (an HF-hub id resolves to a config-only
        # snapshot -- a hash-named dir with just config.json). The measured
        # checkpoint size is unavailable, so instead of silently leaving every
        # family at 2 B/param (BF16 -- which double-counts an FP8/INT4
        # checkpoint and produces a spurious "does not fit / 0 KV"), derive the
        # quantized-family bytes/param AUTHORITATIVELY from the checkpoint's own
        # quantization_config. This makes a hash-named snapshot size IDENTICALLY
        # to the same repo given as a local path.
        cfg_bpp = _config_quant_bpp(cfg, is_moe=self.num_experts > 0)
        if cfg_bpp:
            for name, bpp in cfg_bpp.items():
                families[name].bytes_per_param = bpp
        return families

    def _shard_fractions(self, shard: str, mlp_vector: List[int]) -> List[float]:
        from sglang.srt.distributed.utils import partition_units

        n = self.tp_size
        if shard == "even":
            return [1.0 / n] * n
        if shard == "replicated":
            return [1.0] * n
        if shard == "solo_host":
            # Whole family resident on the solo draft host, nothing anywhere
            # else. Only produced when solo_active (see _build_families), so
            # the split path never reaches this branch.
            return [1.0 if r == self.solo_rank else 0.0 for r in range(n)]
        if shard == "attn":
            grid = self.attn_units
            if grid < n:
                # Replicated-KV regime (#116 / uneven-TP head geometry): fewer
                # KV heads than ranks. Stock even-TP caps TP at the KV-head
                # count, but the fork REPLICATES the KV heads across ranks and
                # shards the token axis (uneven DCP), so >kv_heads ranks are
                # valid. The Q/O projections still shard on the q-head grid
                # (which dominates attn weight; K/V is the small replicated
                # remainder), so we size the shard on the q-head units here
                # instead of crashing in partition_units on 2 KV heads / 3
                # ranks. Only reached when attn_units < tp -- the classic
                # kv_heads>=tp path is untouched (byte-identical).
                grid = max(self.q_heads, n)
            units = partition_units(grid, self.base_plan)
            return [u / grid for u in units]
        if shard in ("gdn", "gdn_base"):
            # vision ("gdn_base") has no own family vector; it follows the
            # base plan on a fine grid -> approximate with exact proportion.
            if shard == "gdn" and self.gdn_units >= n:
                units = partition_units(self.gdn_units, self.base_plan)
                return [u / self.gdn_units for u in units]
            total = float(sum(self.base_plan))
            return [w / total for w in self.base_plan]
        if shard == "mlp":
            units = partition_units(self.mlp_units, mlp_vector)
            return [u / self.mlp_units for u in units]
        raise ValueError(shard)

    def mlp_unit_partition(self, mlp_vector: List[int]) -> List[int]:
        from sglang.srt.distributed.utils import partition_units

        return partition_units(self.mlp_units, mlp_vector)

    def gdn_unit_partition(self) -> List[int]:
        from sglang.srt.distributed.utils import partition_units

        if self.gdn_units >= self.tp_size:
            return partition_units(self.gdn_units, self.base_plan)
        return [0] * self.tp_size

    def per_rank_weight_bytes(self, mlp_vector: List[int]) -> List[float]:
        totals = [0.0] * self.tp_size
        for fam in self.families.values():
            if fam.params <= 0:
                continue
            fracs = self._shard_fractions(fam.shard, mlp_vector)
            for r in range(self.tp_size):
                totals[r] += fam.bytes * fracs[r]
        return totals

    def per_rank_offloadable_weight_bytes(
        self, mlp_vector: List[int]
    ) -> List[float]:
        """Per-rank weight bytes the fork can serve FROM HOST RAM at runtime:
        the MoE ROUTED-expert stack (expert-offload, #77 — a hot subset stays
        resident, the rest live in a pinned host pool). Deliberately EXCLUDES
        everything that must stay on the GPU to serve: dense MLP, attention,
        embeddings/lm_head, GDN/SSM state, and the always-on shared expert
        (it runs on every token). Returns all-zeros for a dense checkpoint, so
        the offload assessment never claims host-offload for a weight class the
        runtime cannot actually tier."""
        if self.num_experts <= 0:
            return [0.0] * self.tp_size
        fam = self.families.get("mlp")
        if fam is None or fam.params <= 0:
            return [0.0] * self.tp_size
        # Routed-expert share of the mlp family mass (the family also carries
        # the small always-resident shared expert, which does NOT offload).
        moe_i = self.moe_intermediate or self.intermediate
        routed = self.num_experts * 3 * self.hidden * moe_i
        shared = 3 * self.hidden * self.shared_expert_intermediate
        routed_frac = routed / (routed + shared) if (routed + shared) else 1.0
        fracs = self._shard_fractions(fam.shard, mlp_vector)
        return [fam.bytes * fracs[r] * routed_frac for r in range(self.tp_size)]

    def _mamba_pool_bytes(self) -> List[float]:
        """Per-rank mamba/SSM pool bytes (state pool + spec-decode
        intermediate), the M22 "SSM pool moves with GDN units" term. Sized
        like the auto-mamba demand path: slots = ceil(target * ratio * 1.25),
        per-request state scales with the rank's GDN-unit share."""
        if not self.gdn_layers or not self.gdn_units:
            return [0.0] * self.tp_size
        ssm_env = os.environ.get("SGLANG_MAMBA_SSM_DTYPE", "")
        ssm_bytes = 2 if "bfloat16" in ssm_env or "float16" in ssm_env else 4
        heads_per_unit = max(self.gdn_v_heads // max(self.gdn_k_heads, 1), 1)
        state_per_unit_layer = (
            heads_per_unit * self.gdn_v_dim * self.gdn_k_dim * ssm_bytes
        )
        conv_per_unit_layer = (
            (2 * self.gdn_k_dim + heads_per_unit * self.gdn_v_dim)
            * (self.conv_kernel - 1)
            * 2
        )
        per_req_per_unit = self.gdn_layers * (
            state_per_unit_layer + conv_per_unit_layer
        )

        target = self.plan_inputs.max_running_requests or 16
        target = min(target, 48)
        ratio = 5  # MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO(3) + overlap(2)
        slots = math.ceil(target * ratio * 1.25)
        d = self.spec_draft_tokens if self.spec_active else 0
        eff_slots = slots + min(target, slots // ratio) * d

        gdn_units = self.gdn_unit_partition()
        return [per_req_per_unit * u * eff_slots for u in gdn_units]

    # -- capacity prediction ------------------------------------------------

    def _solo_rank_token_capacity(self, free_bytes: List[float]) -> List[float]:
        """Per-rank TARGET-KV token capacity under solo draft placement.

        The split model gives every rank a per-token cell of
        ``t_tgt + t_drf`` (target KV + its draft-KV slice). Under solo that is
        wrong on both kinds of rank:

        * SHADOW ranks hold NO draft KV at all -> their cell is ``t_tgt``, so
          they can hold strictly more target tokens than the split model says.
        * The HOST's draft pool is sized to the GLOBAL context C, not to its
          own token share, because the unsharded draft must attend the whole
          sequence. So its draft-KV cost scales with C, not with ``p_host``.

        With the converged token vector (ratios proportional to capacity) the
        host obeys ``free_h = p_h * t_tgt + C * t_drf`` and ``C = sum(p)``.
        Substituting ``Q = sum of the shadows' p`` gives the closed form

            C   = (free_h + Q * t_tgt) / (t_tgt + t_drf)
            p_h = C - Q

        which reduces to the split expression when the draft KV is zero. This
        is the predictor-side mirror of
        ``pool_configurator.solo_draft_kv_cell_factor``.
        """
        per_layer = self.kv_cell_bytes_per_layer
        if self.cross_active:
            # Cross gate: the NEXTN/MTP rung's KV stays in the SHARED target
            # pool on every rank (split placement), so it belongs to t_tgt;
            # only the DFLASH rung's pool is the C-scaled host term.
            t_tgt = self.kv_cell_bytes
        else:
            t_tgt = per_layer * self.full_layers
        if self.solo_draft_kv_cell_bytes is not None:
            # External draft: its own depth/KV geometry (see __init__).
            t_drf = self.solo_draft_kv_cell_bytes
        else:
            t_drf = self.kv_cell_bytes - t_tgt  # target-MTP draft-KV term
        if t_tgt <= 0:
            return [f / self.kv_cell_bytes for f in free_bytes]
        p = [0.0] * self.tp_size
        q = 0.0
        for r in range(self.tp_size):
            if r == self.solo_rank:
                continue
            p[r] = free_bytes[r] / t_tgt
            q += p[r]
        free_h = free_bytes[self.solo_rank]
        if t_drf <= 0:
            p[self.solo_rank] = free_h / t_tgt
            return p
        ctx = (free_h + q * t_tgt) / (t_tgt + t_drf)
        p[self.solo_rank] = ctx - q
        return p

    def predict_capacity(self, mlp_vector: List[int]) -> dict:
        """Predicted per-rank KV token capacity P_r and max context for a
        candidate MLP vector (the base plan's attention/GDN/DCP splits are
        held fixed -- decode is flat across splits per M22, so they are not
        levers). Same math family as the pool sizing: budget minus weight
        bytes minus mamba pool minus reserves, divided by the full-kv-head
        cell; context C = min_r(P_r/ratio_r) * sum(ratios), whose converged
        optimum is min(sum_r P_r, 64 * min_r P_r)."""
        from sglang.srt.distributed.utils import partition_units

        weights = self.per_rank_weight_bytes(mlp_vector)
        if self.measured is not None:
            # MEASURED budget model (registry-backed, cross/solo planning):
            # every non-weight post is a measured value from the previous
            # boot of this same configuration, not a heuristic constant. Per
            # rank r the bytes fundable for KV (target pool + the host's
            # C-scaled draft cell) are
            #
            #   free_r = device_total/ranks_on_gpu     (physical share, NVML)
            #          - residual_residency_r          (driver-measured
            #                                           catch-all: CUDA ctx,
            #                                           NCCL, graphs,
            #                                           workspaces, frag —
            #                                           everything resident
            #                                           that is not weights,
            #                                           pools, or the draft
            #                                           pool; paused rung
            #                                           tags are correctly
            #                                           absent, their remap
            #                                           need is in
            #                                           required_free)
            #          - (model_weights_r(vec) + bias) (family model, anchored
            #                                           to the measured
            #                                           allocator bytes at the
            #                                           measured vector)
            #          - mamba_aux_pool_r              (measured; the mamba/aux
            #                                           pool follows the BASE
            #                                           plan, which is fixed
            #                                           across MLP candidates)
            #          - required_free_r               (configured safety +
            #                                           measured max paused
            #                                           rung tag)
            #
            # The physical total (NOT budgets_mib) is deliberate: under the
            # measured-budget mode the boot converges its pools onto the real
            # leftover regardless of the heuristic budget/reserve knobs, so
            # the planner must model that converged end state. ASSUMPTION
            # (co-location): a card's total and residency split evenly among
            # co-located ranks; exact for the 1-rank-per-card case.
            free_bytes = []
            for r in range(self.tp_size):
                c = self.measured[r]
                total_share = float(c["device_total_bytes"]) / max(
                    int(c["ranks_on_gpu"]), 1
                )
                free_bytes.append(
                    total_share
                    - float(c["residual_residency_bytes"])
                    - (weights[r] + self.measured_weight_bias[r])
                    - float(c["mamba_aux_pool_bytes"])
                    - float(c["required_free_bytes"])
                )
        else:
            overhead = (
                _PREDICT_OVERHEAD_MIB + _PREDICT_MAMBA_ACT_RESERVE_MIB
            ) * 2**20
            # Solo host only: the draft's CUDA graphs and its own attention
            # workspace live on top of the draft weights and draft KV pool,
            # and neither the generic per-rank overhead above nor the weight
            # families cover them. Left out, the host allocates PAST its
            # budget -- measured +634 MiB at max_running_requests=2 and
            # +2266 MiB at 4 on the reference rig, i.e. it grows with the
            # captured decode batch sizes -- which leaves the card with a few
            # hundred MiB free and OOMs during decode. Charged deliberately
            # conservatively: over-reserving costs a little KV,
            # under-reserving costs the whole server.
            solo_overhead = 0.0
            if self.solo_active:
                mrr = int(self.plan_inputs.max_running_requests or 1)
                solo_overhead = (
                    _SOLO_HOST_WORKSPACE_MIB
                    + _SOLO_HOST_GRAPH_MIB_PER_REQ * max(mrr, 1)
                ) * 2**20
            free_bytes = []
            for r in range(self.tp_size):
                budget = self.budgets_mib[r] * 2**20
                extra = (
                    solo_overhead
                    if (self.solo_active and r == self.solo_rank)
                    else 0.0
                )
                free_bytes.append(
                    budget
                    - weights[r]
                    - self.mamba_pool_bytes[r]
                    - overhead
                    - extra
                )
        if self.solo_active:
            p = self._solo_rank_token_capacity(free_bytes)
        else:
            p = [f / self.kv_cell_bytes for f in free_bytes]
        feasible = all(x >= _PREDICT_MIN_RANK_TOKENS for x in p)
        if feasible:
            ctx = min(sum(p), _PREDICT_TOKEN_UNITS * min(p))
            vec = partition_units(
                _PREDICT_TOKEN_UNITS, [max(int(x), 1) for x in p]
            )
            g = math.gcd(*vec)
            vec = [v // g for v in vec]
        else:
            ctx, vec = 0.0, None
        return {
            "p": p,
            "ctx": ctx,
            "token_vector": vec,
            "feasible": feasible,
            "weights_gib": [w / 2**30 for w in weights],
        }

    def residual_free_mib(
        self,
        mlp_vector: List[int],
        device_total_mib: Sequence[int],
        ranks_on_gpu: Sequence[int],
        kv_token_vector: Sequence[int],
    ) -> List[float]:
        """Per-rank VRAM (MiB) a candidate leaves UNALLOCATED after the pools.

        The capacity model above answers "how many KV tokens could this rank
        fund"; this answers the question the boot actually asks next, "and how
        much memory is still free once the pools have been built". They are
        different numbers, and the difference is what task #264 walked into:
        the KV pool is not sized to a rank's own capacity but to the token
        vector, and the vector's scale is set by the TIGHTEST rank
        (``unit = min_r P_r / ratio_r``). A rank that is not the tightest
        therefore keeps its unused capacity as free VRAM -- and MLP
        concentration spends exactly that slack, because it moves the tight
        rank onto the card being concentrated.

        Two terms, both exact under the plan's own assumptions:

          * ``device_total/ranks - budget`` -- the reserve, i.e. the part of
            the card the budget deliberately never claims;
          * ``(P_r - tokens_r) * kv_cell`` -- capacity the token vector does
            not use.

        Everything the budget DOES claim (weights, mamba pool, the modelled
        per-rank overhead) is allocated, so it is not free and is not counted
        here. The absolute level is therefore an upper bound -- the CUDA
        context, NCCL buffers and the graph pool live in the reserve too --
        which is why the caller compares it against a DEMAND
        (``derived_rank_auto_reserve_mib``) rather than against zero, and
        only ever counts a candidate that pushes a rank from above that
        demand to below it.
        """
        pred = self.predict_capacity(list(mlp_vector))
        p = pred["p"]
        vec = [max(int(v), 1) for v in kv_token_vector]
        unit = min(x / v for x, v in zip(p, vec))
        out = []
        for r in range(self.tp_size):
            reserve = device_total_mib[r] / max(int(ranks_on_gpu[r]), 1) - (
                self.budgets_mib[r]
            )
            slack = (p[r] - unit * vec[r]) * self.kv_cell_bytes / 2**20
            out.append(reserve + max(slack, 0.0))
        return out

    # -- speed prediction ---------------------------------------------------

    def streamed_bytes(self, mlp_vector: List[int]) -> List[float]:
        """Per-rank weight bytes STREAMED per decode token (bs=1 decode is
        weight-bandwidth-bound; replicated families stream on every rank)."""
        totals = [0.0] * self.tp_size
        for fam in self.families.values():
            if fam.params <= 0:
                continue
            fracs = self._shard_fractions(fam.shard, mlp_vector)
            for r in range(self.tp_size):
                totals[r] += fam.bytes * fracs[r]
        return totals

    def _mlp_unit_share(self, total_streamed: float) -> float:
        """Streamed-byte-share contributed by ONE MLP unit (the granularity
        of a single representable concentration step). Used to size the
        coarse-grid headroom in the decode-knee guard."""
        if total_streamed <= 0 or self.mlp_units <= 0:
            return 0.0
        return (self.families["mlp"].bytes / self.mlp_units) / total_streamed

    def decode_bw_basis(
        self,
        membw_gbs: Sequence[float],
        gemv_gbs: Optional[Sequence[float]] = None,
    ) -> Tuple[List[float], float, str]:
        """Pick the divisor for the decode roofline, and SAY which one.

        Returns ``(rates, exponent, basis)``. The preferred basis is the
        probe's decode-shaped GEMV rate: it is measured on this rig and
        already carries part of the compression between a card's streaming
        peak and what it reaches reading weights (see
        ``_PREDICT_DECODE_GEMV_RESIDUAL``). Falling back to the streaming peak
        is a real loss of information, so every reason to fall back is a
        named, reported condition -- never a silent max().

        Fallback conditions, per rank:
          * no GEMV rate at all (profile predates PROFILE_VERSION 2);
          * GEMV below ``_PROBE_GEMV_SATURATION_FLOOR`` of the streaming
            rate -- not bandwidth-bound, so it measures something else;
          * GEMV above ``_PROBE_GEMV_CACHE_CEILING`` of it -- served from
            cache, so it measures something else.
        A single unusable rank falls the whole rig back, because the roofline
        consumes RATIOS between ranks: mixing a GEMV rate on one rank with a
        streaming rate on another would compare two different quantities."""
        peak = [max(float(b), 1e-9) for b in membw_gbs]
        beta_peak = self.calibration.peak_compression
        if gemv_gbs is None or len(gemv_gbs) != len(peak):
            return (
                peak,
                beta_peak,
                "streaming peak (no GEMV rate in the hardware profile; "
                "re-probe with SGLANG_PERF_REPROBE=1 to measure it)",
            )
        gemv = [float(g or 0.0) for g in gemv_gbs]
        for r, (g, p) in enumerate(zip(gemv, peak)):
            frac = g / p
            if frac < _PROBE_GEMV_SATURATION_FLOOR:
                return (
                    peak,
                    beta_peak,
                    f"streaming peak (rank {r}: GEMV {g:.0f} GB/s is "
                    f"{frac * 100:.0f} % of its streaming {p:.0f} GB/s, below "
                    f"the {_PROBE_GEMV_SATURATION_FLOOR * 100:.0f} % "
                    f"saturation floor -- the probe is not bandwidth-bound on "
                    f"this architecture and its rate is not a weight-read "
                    f"rate)",
                )
            if frac > _PROBE_GEMV_CACHE_CEILING:
                return (
                    peak,
                    beta_peak,
                    f"streaming peak (rank {r}: GEMV {g:.0f} GB/s exceeds its "
                    f"streaming {p:.0f} GB/s, so it is reading from cache "
                    f"rather than DRAM and is not a weight-read rate)",
                )
        return gemv, self.calibration.gemv_residual, "measured decode GEMV rate"

    def effective_decode_bw(
        self,
        membw_gbs: Sequence[float],
        gemv_gbs: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Per-rank bandwidth actually reached while streaming a weight shard
        at batch size 1.

        Divisor and exponent come from ``decode_bw_basis``; the exponent
        supplies only what the probe could not measure. Returned in arbitrary
        units: only ratios between ranks are ever consumed."""
        rates, beta, _ = self.decode_bw_basis(membw_gbs, gemv_gbs)
        return [max(float(b), 1e-9) ** beta for b in rates]

    def decode_round_time(
        self,
        mlp_vector: List[int],
        membw_gbs: Sequence[float],
        gemv_gbs: Optional[Sequence[float]] = None,
    ) -> float:
        """Relative bs=1 decode round time: the lockstep max over ranks of
        (streamed weight bytes / effective bandwidth). Only ratios between
        candidates are consumed."""
        bw = self.effective_decode_bw(membw_gbs, gemv_gbs)
        streamed = self.streamed_bytes(list(mlp_vector))
        return max(s / b for s, b in zip(streamed, bw))

    def per_rank_decode_times(
        self,
        mlp_vector: List[int],
        membw_gbs: Sequence[float],
        gemv_gbs: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Each rank's OWN bs=1 weight-streaming time for one decode round.

        The vector ``decode_round_time`` takes the max of, in the same
        arbitrary-but-consistent units. Exposed for the energy objective
        under ``--rank-perf-tune dec`` (#434), which needs every rank's busy
        term: the round lasts as long as the slowest rank and the others draw
        idle-to-active in proportion to how much of it they were busy --
        exactly what ``per_rank_prefill_compute_times`` supplies on the
        prefill side. Pure arithmetic over the profile's own rates; no
        probing.
        """
        bw = self.effective_decode_bw(membw_gbs, gemv_gbs)
        streamed = self.streamed_bytes(list(mlp_vector))
        return [s / b for s, b in zip(streamed, bw)]

    def decode_cost_percent(
        self,
        mlp_vector: List[int],
        membw_gbs: Sequence[float],
        gemv_gbs: Optional[Sequence[float]] = None,
    ) -> float:
        """Predicted change of the bs=1 decode STEP time against the base
        plan, in percent. Positive = the candidate makes decode slower.

        This exists because a guard that only answers yes/no is how a vector
        that costs 16.5 % of the decode step passed review in silence. Even
        when the verdict is wrong the number is actionable, so it is reported
        for every candidate, accepted or rejected.

        The weight term moves with the MLP split; the rest of the step (KV
        attention, draft+verify, collectives, launch overhead) does not, and
        is held at ``_PREDICT_DECODE_NONWEIGHT_FRACTION`` of the base step so
        the percentage refers to the step a user actually waits for rather
        than to the weight term alone."""
        base = self.decode_round_time(self.base_plan, membw_gbs, gemv_gbs)
        if base <= 0:
            return 0.0
        ratio = self.decode_round_time(mlp_vector, membw_gbs, gemv_gbs) / base
        f = self.calibration.nonweight_fraction
        return (f + (1.0 - f) * ratio - 1.0) * 100.0

    def decode_knee_ok(
        self,
        mlp_vector: List[int],
        membw_gbs: Sequence[float],
        tol: float = _PREDICT_DECODE_KNEE_TOL,
        gemv_gbs: Optional[Sequence[float]] = None,
    ) -> bool:
        """Decode-knee guard (M20/M22/M23/#216-measured): decode throughput is
        FLAT while every rank's share of the streamed weight bytes stays at
        or below its share of the rig's EFFECTIVE decode bandwidth; pushing a
        rank past that knee makes it the decode lockstep bottleneck (M23:
        16,1,2 = 56.5% bytes on the 51.9%-membw 5090 -> decode -4.8%) and
        the prefill model's extra predicted gain does NOT materialize
        (measured +10.0%, identical to the knee-exact C6 vector).

        Effective, not probed peak. The two differ by an architecture- and
        quant-dependent factor that does not cancel across a mixed rig, and
        taking peak for achieved is what let 6,1,1 through at a measured
        +16.5 % decode cost. The effective rate is built from the probe's
        decode-shaped GEMV where it is usable and from the streaming peak
        otherwise -- see ``decode_bw_basis``, which names which of the two it
        used and why.

        The per-rank share is computed from the ACTUAL integer unit partition
        (``mlp_unit_partition`` via ``streamed_bytes``), never from the wish
        ratio, so a coarse grid's rounding is taken at face value. Because
        coarse grids (FP8 dense ~136 units, GGUF K-quant ~68) realize the
        split in large steps, the nearest representable vector can sit above
        the measured knee while its whole-model share still reads just under
        the bandwidth share (M27d: FP8 4,1,1 = 51.5% share < 51.9% membw, yet
        decode -14/-24%). On such coarse grids we therefore require
        ``_PREDICT_KNEE_COARSE_HEADROOM_UNITS`` unit-steps of headroom below
        the knee and round DOWN to the last safe vector; fine grids keep the
        exact bandwidth-share ceiling. A rank that only sheds/keeps bytes vs
        the base plan can never become the knee and is skipped."""
        ok, _ = self.decode_knee_detail(mlp_vector, membw_gbs, gemv_gbs)
        return ok

    def decode_knee_detail(
        self,
        mlp_vector: List[int],
        membw_gbs: Sequence[float],
        gemv_gbs: Optional[Sequence[float]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Like ``decode_knee_ok`` but also returns a human-readable reason
        for the first violating rank, naming the unit granularity (for the
        optimizer's per-candidate log line)."""
        cand = self.streamed_bytes(mlp_vector)
        base = self.streamed_bytes(self.base_plan)
        total = sum(cand)
        # EFFECTIVE, not peak: the ceiling has to be the bandwidth a rank
        # really reaches on this workload, or it certifies concentration the
        # rank cannot absorb (see decode_bw_basis).
        bw_eff = self.effective_decode_bw(membw_gbs, gemv_gbs)
        bw_total = sum(bw_eff)
        coarse = self.mlp_units < _PREDICT_KNEE_COARSE_UNITS
        unit_share = self._mlp_unit_share(total)
        headroom = (
            _PREDICT_KNEE_COARSE_HEADROOM_UNITS * unit_share if coarse else 0.0
        )
        for r in range(self.tp_size):
            if cand[r] <= base[r] * (1.0 + 1e-6):
                continue  # rank sheds or keeps bytes: cannot become the knee
            achievable = cand[r] / total if total else 0.0
            membw_share = bw_eff[r] / bw_total if bw_total else 0.0
            limit = membw_share - headroom
            if achievable > limit:
                # requested = share the wish ratio (continuous, no rounding)
                # aimed for on this rank; achievable = the real partition's
                # share; naming both makes the coarse-grid overshoot explicit.
                requested = self._requested_mlp_share(mlp_vector, r, total)
                grain = (
                    f"coarse {self.mlp_units}-unit MLP grid, "
                    f"{_PREDICT_KNEE_COARSE_HEADROOM_UNITS}-unit headroom"
                    if coarse
                    else f"fine {self.mlp_units}-unit MLP grid"
                )
                reason = (
                    f"unit granularity ({grain}): rank {r} requested "
                    f"{requested * 100:.1f}% -> achievable {achievable * 100:.1f}% "
                    f"of streamed weight bytes exceeds EFFECTIVE membw share "
                    f"{membw_share * 100:.1f}% (peak share "
                    f"{membw_gbs[r] / max(sum(membw_gbs), 1e-9) * 100:.1f}%, "
                    f"safe ceiling {limit * 100:.1f}%); predicted decode step "
                    f"{self.decode_cost_percent(mlp_vector, membw_gbs, gemv_gbs):+.1f}%"
                )
                return False, reason
        return True, None

    def _requested_mlp_share(
        self, mlp_vector: List[int], rank: int, total_streamed: float
    ) -> float:
        """Whole-model streamed-byte share rank `rank` would have if the MLP
        family used the CONTINUOUS wish fraction (no integer unit rounding).
        Contrasted with the achievable (real-partition) share to expose the
        coarse-grid overshoot in the log."""
        if total_streamed <= 0:
            return 0.0
        wish = mlp_vector[rank] / sum(mlp_vector)
        acc = 0.0
        for name, fam in self.families.items():
            if fam.params <= 0:
                continue
            if fam.shard == "mlp":
                acc += fam.bytes * wish
            else:
                acc += fam.bytes * self._shard_fractions(fam.shard, mlp_vector)[rank]
        return acc / total_streamed

    def _prefill_sharded_time(
        self,
        mlp_vector: List[int],
        gemm_tflops: List[float],
        min_link_gbs: Optional[float],
        family_tflops: Optional[Dict[str, List[float]]] = None,
    ) -> float:
        """The shard-PROPORTIONAL part of the prefill step: lockstep compute
        max over ranks (per-token flops ~ 2 x sharded params, the
        param-proxy) plus the ring-all-reduce term over the narrowest
        link.

        ``family_tflops`` (#287, delegated by #324) maps a WEIGHT family name
        to its own per-rank compute rate. When given, the per-rank time is
        ``sum_fam(2 * p_fam / r_fam)`` -- each family's params divided by the
        rate of the lane THAT family runs on this rank. The pre-#287 scalar
        arithmetic (``sum_fam(p_fam) / r``) is exact only while every family
        runs one lane per rank; on a MIXED_PRECISION checkpoint the same card
        runs e.g. MLP on Marlin (216 TFLOPS) and attn/GDN on native fp8
        (566.88), a 2.62x band that a summed-params-over-one-rate model
        mis-times. ``None`` (every single-scheme checkpoint) keeps the scalar
        arithmetic byte-identical -- the two differ in the last bits even for
        equal rates, because float addition of times is not float division of
        a param sum.
        """
        # Per-rank compute times, then their max. Factored out of this method
        # rather than inlined so the ENERGY objective (#350 p4) can read every
        # rank's own time -- a rank that finishes early still draws power
        # while it waits, so the max alone cannot price joules. The max over
        # the ranks in ascending order is the identical float to the running
        # `max(t_comp, t)` this replaced, so the throughput path is unchanged.
        # Called through the CLASS, not through ``self``: the arithmetic is
        # duck-typed on (tp_size, families, _shard_fractions) and several
        # tests bind this method to a stand-in that provides exactly those
        # and nothing else. Going through the class keeps every such caller
        # working, which is what lets the factoring stay a pure refactor.
        t_comp = max(
            PerfCostModel.per_rank_prefill_compute_times(
                self, mlp_vector, gemm_tflops, family_tflops
            ),
            default=0.0,
        )
        # Two all-reduces of H bf16 per layer per token, ring factor
        # 2(N-1)/N, bounded by the narrowest participating link.
        #
        # ``min_link_gbs=None`` means no pair matrix was measured: the term is
        # NOT priced and the caller holds a compute-only time (#359). It used
        # to be ``max(min_link_gbs, 0.1)``, a silent floor that rescued any
        # value below it -- including the 1e-3 GB/s placeholder the key solver
        # passed, which therefore never reached this arithmetic at all. A
        # non-positive link is now a caller error, loudly.
        n = self.tp_size
        if n <= 1 or min_link_gbs is None:
            return t_comp
        if min_link_gbs <= 0.0:
            raise ValueError(
                f"min_link_gbs must be a measured positive rate, got "
                f"{min_link_gbs!r}. Pass None to price the compute term only "
                "when no pair matrix exists; there is no stand-in constant."
            )
        ar_bytes = self.n_layers * 2 * self.hidden * 2
        t_comm = ar_bytes * 2 * (n - 1) / n / (min_link_gbs * 1e9)
        return t_comp + t_comm

    def per_rank_prefill_compute_times(
        self, mlp_vector, gemm_tflops, family_tflops=None
    ) -> List[float]:
        """Each rank's OWN prefill GEMM time for one lockstep round (s).

        The vector ``_prefill_sharded_time`` takes the max of. Exposed
        because the energy objective (#350) needs every rank's term: the
        round lasts as long as the slowest rank, and the others draw
        idle-to-active in proportion to how much of it they were busy.
        Pure arithmetic over the same family shard fractions; no probing.
        """
        out: List[float] = []
        if family_tflops:
            denom = 1e12 * _PREDICT_GEMM_EFF
            for r in range(self.tp_size):
                t = 0.0
                for name, fam in self.families.items():
                    if fam.params <= 0 or fam.shard == "replicated":
                        continue
                    params_fam_r = (
                        fam.params * self._shard_fractions(fam.shard, mlp_vector)[r]
                    )
                    rates = family_tflops.get(name)
                    rate_r = rates[r] if rates else gemm_tflops[r]
                    t += 2.0 * params_fam_r / (rate_r * denom)
                out.append(t)
        else:
            for r in range(self.tp_size):
                params_r = 0.0
                for fam in self.families.values():
                    if fam.params <= 0 or fam.shard == "replicated":
                        continue
                    params_r += (
                        fam.params * self._shard_fractions(fam.shard, mlp_vector)[r]
                    )
                out.append(
                    2.0 * params_r / (gemm_tflops[r] * 1e12 * _PREDICT_GEMM_EFF)
                )
        return out

    def prefill_time_model(
        self,
        mlp_vector: List[int],
        gemm_tflops: List[float],
        min_link_gbs: Optional[float],
        family_tflops: Optional[Dict[str, List[float]]] = None,
    ) -> float:
        """Relative prefill step time. Only ratios between candidates are
        consumed. ``family_tflops`` switches the sharded term to the
        per-family ``sum(p/r)`` arithmetic (see ``_prefill_sharded_time``);
        ``None`` is the byte-identical scalar path.

        ``min_link_gbs=None`` prices the compute term only, for a rig whose
        pair matrix was never measured. The omitted collective term is
        split-invariant, so the ORDER over candidates is the order at any link
        rate -- an argmax stays answerable. A RATIO between two of these times
        is not: it drops an additive constant from both sides and overstates
        the move, so it must not be compared against a threshold. See
        ``cost_model.ABSENT_LINK_COMPUTE_ONLY_REASON``.

        Two terms: the shard-proportional part (``_prefill_sharded_time``:
        GEMM compute at the probe rate + per-layer all-reduce) and a split-
        INVARIANT part -- eager per-layer launch overhead (the measured boots
        report ``prefill.backend='disabled'``: no captured prefill graph) and
        the non-GEMM ops, none of which shrink when MLP units leave a rank.
        The invariant part is charged as
        ``prefill_invariant_fraction`` of the BASE plan's step (see the
        constant's derivation + refit formula), so it is a constant across
        candidates: ranking is untouched, but the reported gains no longer
        pretend the whole step scales with the shard. Without it the model
        over-predicted every measured concentration gain by x1.4-1.8
        (predicted +21.6 % for 6,1,1 where +13.0 % was measured)."""
        f = self.calibration.prefill_invariant
        t_sharded = self._prefill_sharded_time(
            mlp_vector, gemm_tflops, min_link_gbs, family_tflops
        )
        if f <= 0.0:
            return t_sharded
        t_base = self._prefill_sharded_time(
            self.base_plan, gemm_tflops, min_link_gbs, family_tflops
        )
        return t_sharded + f / (1.0 - f) * t_base


# ---------------------------------------------------------------------------
# The optimizer: candidate MLP vectors -> floor filter -> objective.
# ---------------------------------------------------------------------------


def _gcd_reduce(vec: Sequence[int]) -> Tuple[int, ...]:
    g = math.gcd(*vec)
    return tuple(v // g for v in vec)


def _mlp_candidates(
    model: PerfCostModel, scores: List[float], base_plan: List[int]
) -> List[Tuple[int, ...]]:
    """Small deduplicated candidate set of MLP unit vectors.

    Two ladders: (a) score-proportional concentration at several strengths
    (score^alpha), (b) the compute-balance solution (assign each rank MLP
    mass until work_r/score_r equalizes, the analytic prefill optimum),
    rounded at several integer resolutions."""
    n = model.tp_size
    cands: List[Tuple[int, ...]] = []
    smax = max(scores)

    for alpha in (0.5, 1.0, 1.5, 2.0):
        for k in (3, 5, 8):
            vec = tuple(
                max(1, round(k * (s / smax) ** alpha)) for s in scores
            )
            cands.append(_gcd_reduce(vec))

    # Balance ladder: fixed (non-MLP) work per rank + m_r ~ score share.
    fixed = [0.0] * n
    mlp_mass = 0.0
    for fam in model.families.values():
        if fam.params <= 0 or fam.shard == "replicated":
            continue
        if fam.shard == "mlp":
            mlp_mass += fam.params
            continue
        fr = model._shard_fractions(fam.shard, base_plan)
        for r in range(n):
            fixed[r] += fam.params * fr[r]
    total = sum(fixed) + mlp_mass
    s_sum = sum(scores)
    m = [max(total * s / s_sum - fixed[r], 0.0) for r, s in enumerate(scores)]
    m_sum = sum(m)
    if m_sum > 0:
        for k in (6, 10, 16):
            vec = tuple(max(1, round(k * x / max(m))) for x in m)
            cands.append(_gcd_reduce(vec))

    base = _gcd_reduce(base_plan)
    out, seen = [], set()
    for c in cands:
        if c not in seen and c != base and len(set(c)) > 1:
            seen.add(c)
            out.append(c)
    return out


#: Gate names in the order the per-candidate verdict applies them.
_GATE_ORDER = ("infeasible", "unbootable", "floor", "knee", "no gain")

#: ``--rank-perf-tune`` targets that plan the PHASE-OPTIMAL recipe (#357):
#: one weight vector per phase instead of one vector for the whole server.
#: ``phase-prefill`` installs the concentrated prefill vector,
#: ``phase-decode`` installs the VRAM-auto split (the measured decode
#: optimum); both arms report the same pair, so one plan run states the whole
#: recipe.
_PHASE_PREFILL = "phase-prefill"
_PHASE_DECODE = "phase-decode"
_PHASE_TUNES = (_PHASE_PREFILL, _PHASE_DECODE)


def planner_corridor_mib() -> int:
    """#330's absolutely-free corridor, in MiB, as the planner prices it.

    AUDIT_434 class ``POLICY``: a deliberate rig-independent rule ("at least
    this much VRAM stays unallocated on every card"), not a constant fitted on
    the reference rig -- it is the same number on a 3080 and on a 5090, and it
    is a choice about how much room a boot must leave, not a measurement of
    anything. There is exactly one definition of it on this rig,
    ``registry.ledger.DEFAULT_CORRIDOR_BYTES`` (#330), and this reads that one
    rather than restating 400; the ledger daemon already exposes it as
    ``--corridor-mib``, and the override seam for the planner's use of it is
    ``SGLANG_PLANNER_CORRIDOR_MIB``.

    Why the planner needs it at all: the fundability gate compares a rank's
    predicted residual free VRAM against the derived reserve DEMAND, i.e.
    against the modelled non-budget posts. Clearing that demand exactly means
    a boot that allocates every last modelled byte and leaves nothing -- the
    #424 window measured 87 MiB free on the 5090 at one such operating point
    and 286-316 MiB at another, and both were treated as corridor breaches by
    hand. Pricing the corridor here is what turns that hand judgement into
    something the plan log states before the boot.
    """
    from sglang.srt.environ import envs
    from sglang.srt.registry.ledger import DEFAULT_CORRIDOR_BYTES, MIB

    override = envs.SGLANG_PLANNER_CORRIDOR_MIB.get()
    if override is not None:
        return max(int(override), 0)
    return int(DEFAULT_CORRIDOR_BYTES // MIB)


def _phase_solve_owns_kv_ratio(server_args, model, tune: str) -> bool:
    """True when the phase solve's own matched KV token vector becomes the
    boot's DCP token vector (#435).

    The defect this names: the optimizer derives a capacity-MATCHED token
    vector for every candidate, gates the candidate on the context that
    vector funds, and then -- before #435 -- let the boot resolve the coupled
    KV ratio from the VRAM-BUDGET split instead. The two agree only while the
    weight vector is the base plan, which is precisely what the phase-prefill
    arm exists to change.

    Conditions, each for a reason:

    ``tune in _PHASE_TUNES``
        Only the phase recipe concentrates weight mass away from the budget
        proportion, which is what makes the two vectors diverge. Every other
        target keeps the previous behavior byte-identically. ``phase-decode``
        cannot reach the install (it solves the plain VRAM-auto split and
        clears ``chosen``), so in practice this is the prefill arm; it is
        written as the family because the condition is about the recipe, not
        about which of its two arms happens to install a vector today.
    ``not model.solo_active``
        Draft-solo placement has its own, older seeding rule (the host's
        unsharded draft weights and globally-sized draft KV pool are not in
        the budget estimate). It is left exactly as it was.
    ``rank_kv_ratio == 'coupled'``
        An explicit ``--rank-kv-ratio`` -- a pinned vector or a derived mode
        string -- always wins (#88 family-flag precedence). ``'coupled'`` is
        the default, i.e. "no opinion expressed"; an explicit
        ``--rank-kv-ratio coupled`` is deliberately indistinguishable from it,
        the same convention ``_check_perf_flags`` uses for the ``dec`` and
        ``maxkv`` targets.
    """
    return (
        tune in _PHASE_TUNES
        and not model.solo_active
        and getattr(server_args, "rank_kv_ratio", "coupled") == "coupled"
    )


def _binding_gate(entry, knee_binding: bool = True) -> str:
    """Which gate rejected this candidate, in verdict precedence order.

    ``knee_binding`` is False in the phase-optimal modes (#357), where the
    decode-knee verdict is reported but does not reject: that vector serves
    the PREFILL phase and the decode phase runs the companion split, so a
    decode cost it never pays cannot be the gate that refuses it."""
    _cand, pred, gain, floor_ok, knee_ok, _reason, _res, unfundable = entry
    if not pred["feasible"]:
        return "infeasible"
    if unfundable is not None:
        return "unbootable"
    if not floor_ok:
        return "floor"
    if not knee_ok and knee_binding:
        return "knee"
    return "no gain" if gain <= 0 else "accepted"


def _no_lever_lines(
    tune: str,
    loose: float,
    results,
    knee_binding: bool = True,
    metric: str = "prefill",
) -> List[str]:
    """The refusal, when no candidate survives the gates (#265).

    Before this, the optimizer printed one generic sentence that named the
    context floor and recommended raising ``--rank-perf-loose-ctx-percent``
    whatever the actual binding gate was. Two things were wrong with that.
    It sent the reader to a knob that cannot help when the binding gate is
    the decode-knee guard or fundability -- with fundability it buys an OOM
    (#264) -- and for ``--rank-perf-tune enc`` it hid the only fact worth
    reporting: enc's single lever is MLP concentration, so "every
    concentrated candidate rejected" means enc has no lever at this
    operating point, not that enc optimized and found nothing.

    The refusal therefore names the tally per gate, the best forfeited
    candidate, and only mentions the loose-ctx knob when the floor is what
    actually binds.
    """
    if not results:
        return [
            f"tune={tune}: no representable concentration candidate on this "
            "MLP grid -- keeping the plain VRAM-auto split."
        ]
    gates = [_binding_gate(e, knee_binding) for e in results]
    tally = ", ".join(
        f"{g} {gates.count(g)}" for g in _GATE_ORDER if gates.count(g)
    )
    best = max(results, key=lambda e: e[2])
    best_gate = _binding_gate(best, knee_binding)
    lever = (
        "enc has no effective lever at this operating point"
        if tune == "enc"
        else (
            "the phase-optimal prefill arm has no lever at this operating "
            "point"
            if tune in _PHASE_TUNES
            else (
                # #434: the answer for 'dec' is now SOLVED on this rig's own
                # bandwidth profile rather than asserted from M22's
                # reference-rig measurement, so the refusal has to read as a
                # result: on this profile the weight split is flat for
                # decode. The other decode lever (--rank-kv-ratio speed) is
                # selected independently and is unaffected by this verdict.
                "the weight split is FLAT for decode on this rig's measured "
                "bandwidth profile -- the VRAM-auto split already sits at "
                "the decode optimum this MLP grid can represent"
                if tune == "dec"
                else "no candidate survives the gates"
            )
        )
    )
    head = (
        f"tune={tune}: {lever}. {len(results)} concentration candidates "
        f"evaluated, none accepted ({tally}). The best of them, "
        f"{','.join(map(str, best[0]))}, would have predicted "
        f"{best[2] * 100:+.1f}% {metric} and is rejected by: {best_gate}. "
        "Keeping the plain VRAM-auto split."
    )
    tail = {
        "floor": (
            "The floor is what binds: raise --rank-perf-loose-ctx-percent "
            f"(now {loose:g}) to trade predicted context for {metric} speed."
        ),
        "knee": (
            "The decode-knee guard is what binds: the candidate would make "
            "the strong rank the decode lockstep pacer. "
            "--rank-perf-loose-ctx-percent does not move this gate; it is a "
            "speed/speed trade, not a context trade."
        ),
        "unbootable": (
            "Fundability is what binds: the candidate leaves a rank below "
            "its derived reserve demand, i.e. it does not boot rather than "
            "serving less context. --rank-perf-loose-ctx-percent cannot "
            "accept it -- raise --rank-auto-reserve-mib on the named GPU (the "
            "#264 case: 6,1,1 needed 4500 where the base ran at 3000)."
        ),
        "infeasible": (
            "The candidates are infeasible outright: at least one rank falls "
            "below the minimum viable token capacity. Neither "
            "--rank-perf-loose-ctx-percent nor a bigger reserve helps; the "
            "budgets are too small for this concentration."
        ),
        "no gain": (
            f"No gate binds -- the model simply predicts no {metric} gain "
            "from concentration here, so the VRAM-auto split already sits at "
            f"the {metric} optimum this rig can represent."
        ),
    }.get(best_gate)
    return [head] + ([tail] if tail else [])


@dataclasses.dataclass
class PerfDecision:
    chosen_vector: Optional[List[int]]  # None = keep the plain auto split
    log_lines: List[str]


def _tp_drop_recommendation(
    server_args, profile: dict, gpus: List[dict], model: PerfCostModel
) -> Optional[str]:
    """Stage-1 TP-degree reduction RECOMMENDATION (log only, never applied):
    when one GPU sits behind a clearly narrower link and the remaining
    budgets still fit the weights, dropping it bought +55-76% prefill /
    +25-30% concurrent at -72% context in the M22 measurements."""
    if server_args.tp_size < 3 or not profile.get("links"):
        return None
    by_uuid = {g["uuid"]: g for g in gpus}
    rank_gpu = server_args.rank_gpu_id
    used_uuids = []
    for gid in rank_gpu:
        match = [g for g in gpus if g["cuda_index"] == gid]
        if not match:
            return None
        used_uuids.append(match[0]["uuid"])
    unique = sorted(set(used_uuids))
    if len(unique) < 3:
        return None

    def pair_bw(u1: str, u2: str) -> Optional[float]:
        e = profile["links"].get("|".join(sorted([u1, u2])))
        return e.get("p2p_gbs") if e else None

    best_link = 0.0
    per_gpu_best: Dict[str, float] = {}
    for u in unique:
        bws = [pair_bw(u, v) for v in unique if v != u]
        bws = [b for b in bws if b]
        if not bws:
            return None
        per_gpu_best[u] = max(bws)
        best_link = max(best_link, max(bws))
    weakest = min(per_gpu_best, key=per_gpu_best.get)
    if per_gpu_best[weakest] >= _TP_DROP_LINK_FRACTION * best_link:
        return None

    # Fit check: total weights (even split irrelevant -- total conserved)
    # must fit into the REMAINING ranks' budgets with headroom.
    total_weights = sum(model.per_rank_weight_bytes(list(model.base_plan)))
    keep_budget = sum(
        model.budgets_mib[r] * 2**20
        for r in range(model.tp_size)
        if used_uuids[r] != weakest
    )
    if total_weights > _TP_DROP_FIT_FACTOR * keep_budget:
        return None

    keep_ids = [
        str(rank_gpu[r]) for r in range(model.tp_size) if used_uuids[r] != weakest
    ]
    name = by_uuid[weakest]["name"]
    return (
        f"RECOMMENDATION (not applied): GPU {by_uuid[weakest]['cuda_index']} "
        f"({name}) sits behind the narrowest link "
        f"({per_gpu_best[weakest]:.1f} GB/s vs best {best_link:.1f} GB/s). "
        f"Dropping it (--tp-size {len(keep_ids)} --rank-gpu-id "
        f"{','.join(keep_ids)}) measured +55-76% prefill / +25-30% "
        "concurrent at -72% max context in the M22 feasibility matrix. "
        "TP degree is never changed silently; re-launch with those flags "
        "to take this trade."
    )


def _objective_is_energy(server_args) -> bool:
    """True when the boot was asked to plan for joules (#350 phase 4)."""
    from sglang.srt.planner.objective import Objective, resolve_objective

    return resolve_objective(server_args) is Objective.ENERGY


def _boot_energy_model(server_args, model, lines: List[str]):
    """Power anchors for the boot planner, or a hard failure.

    Fails LOUDLY rather than falling back. A silent throughput boot under
    ``--objective energy`` is exactly the substitution the provenance rule
    forbids, and the fork validates early: an unpriceable rig is a
    configuration error the operator must see at parse time, not a plan whose
    label lies.
    """
    from sglang.srt.planner.objective import boot_energy_anchors

    # Per-rank card identity comes from the CACHED HARDWARE PROFILE, which is
    # the same artifact the rest of the auto-performance path already reads.
    # (Earlier this looked for a `gpu_names` attribute on the cost model; no
    # such attribute exists, so the energy boot could only ever reach the
    # refusal below -- found by running it, #350 validation window.)
    names: List[str] = []
    uuids: List[str] = []
    try:
        profile, _inv = get_cached_hardware_profile()
        gpus = (profile or {}).get("gpus") or {}
        by_index = {}
        for uuid, entry in gpus.items():
            idx = entry.get("cuda_index")
            if idx is not None:
                by_index[int(idx)] = (str(entry.get("name") or ""), str(uuid))
        rank_ids = list(getattr(server_args, "rank_gpu_id", None) or [])
        if not rank_ids:
            rank_ids = sorted(by_index)
        for cuda_idx in rank_ids:
            hit = by_index.get(int(cuda_idx))
            if hit is None:
                names = []
                break
            names.append(hit[0])
            uuids.append(hit[1])
    except Exception:
        names, uuids = [], []
    if not names:
        raise ValueError(
            "--objective energy: the boot planner has no per-rank card "
            "identities, so it cannot look up power anchors. This is a "
            "planning-path gap, not a user error -- plan through "
            "/api/key_solver with explicit power_anchors, or drop "
            "--objective energy."
        )
    energy_model, notes = boot_energy_anchors(names, uuids or None)
    if energy_model is None:
        raise ValueError(
            "--objective energy cannot be priced on this rig: "
            + "; ".join(notes)
            + ". Run the #149 power calibration, or add the card's TDP to the "
            "card library, or drop --objective energy. Booting the "
            "throughput vector under an energy flag would be a silent "
            "substitution, so the boot is refused instead."
        )
    lines.append(
        f"objective=energy: planning for J/token, {energy_model.provenance.value} "
        f"power anchors ({'; '.join(notes)})"
    )
    return energy_model


def apply_auto_performance(server_args) -> None:
    """Entry point for --rank-tp-ratio auto-performance, called from
    ServerArgs._handle_uneven_tp AFTER the VRAM-auto base split is resolved
    and BEFORE the family-vector validation. Derives --rank-mlp-ratio from
    the hardware profile, subject to the context floor; logs one block with
    every decision input. Never touches the base (attention/GDN/DCP) split.
    """
    lines: List[str] = ["auto-performance (--rank-tp-ratio auto-performance):"]

    # Pin the measured-registry path NOW (parse-time field state) and stash
    # it for the boot-time writer — see measured_kv_budget_cache_path.
    server_args._measured_kv_budget_registry_path = (
        measured_kv_budget_cache_path(server_args)
    )

    def emit():
        logger.info("\n  ".join(lines))

    tune = server_args.rank_perf_tune
    loose = float(server_args.rank_perf_loose_ctx_percent)

    if not isinstance(server_args.rank_tp_ratio, list):
        lines.append(
            "base VRAM-auto split collapsed to the even split (uniform "
            "budgets); the MLP family vector requires an uneven base plan "
            "-- keeping the classic even split unchanged."
        )
        emit()
        return

    base_plan = list(server_args.rank_tp_ratio)
    # Decoupled KV-token ownership (--rank-kv-ratio, task #88): the
    # capacity predictions below (predict_capacity's ctx = min(sum P,
    # 64*min P)) always assume the CONVERGED capacity-optimal token
    # vector. Under 'capacity' that assumption is realized on the first
    # boot (measured install after profiling); under 'coupled' the phase
    # arms seed the predicted match themselves (#435,
    # _phase_solve_owns_kv_ratio) and every other target still needs the
    # SGLANG_UNEVEN_TOKEN_VECTOR restart hint. Except for that seed, the
    # KV ownership vector is chosen independently of the MLP/GEMM vector
    # this optimizer picks.
    if server_args.uneven_kv_flag_active():
        kv_mode = server_args.rank_kv_ratio
        lines.append(
            "KV-token ownership decoupled (--rank-kv-ratio "
            f"{','.join(map(str, kv_mode)) if isinstance(kv_mode, list) else kv_mode}): "
            "the context floor below is evaluated against the converged "
            "weighted-DCP optimum, which this mode "
            + (
                "realizes on the first boot (measured install after "
                "profiling)."
                if server_args.uneven_kv_derived_mode()
                else "pins explicitly."
            )
        )
    budgets = server_args.rank_gpu_memory_mib
    budgets = (
        list(budgets)
        if isinstance(budgets, list)
        else [budgets] * server_args.tp_size
    )

    # Pin path: an explicit vector (flag or env) skips probe + optimizer.
    from sglang.srt.environ import envs

    pinned = server_args.rank_mlp_ratio
    env_pin = envs.SGLANG_UNEVEN_MLP_VECTOR.get()
    if pinned is not None or env_pin:
        lines.append(
            f"MLP vector PINNED ({'SGLANG_UNEVEN_MLP_VECTOR=' + env_pin if env_pin else '--rank-mlp-ratio ' + ','.join(map(str, pinned))}); "
            "hardware probe and optimizer skipped (pin path)."
        )
        emit()
        return

    profile, source, gpus = get_hardware_profile()
    if profile is None:
        lines.append(
            "hardware profile unavailable (probe failed) -- keeping the "
            "plain VRAM-auto split; fix the probe or pin --rank-mlp-ratio."
        )
        emit()
        return

    lines.append(f"hardware profile: {source}")
    uuid_by_idx = {g["cuda_index"]: g["uuid"] for g in gpus}
    rank_scores_gemm: List[float] = []
    rank_scores_bw: List[float] = []
    # Decode-shaped GEMV rate per rank, or None if ANY rank's profile entry
    # lacks it (a v1 cache alongside a v2 one cannot happen -- the version is
    # part of the cache key -- but a hand-edited profile can).
    rank_scores_gemv: Optional[List[float]] = []
    rank_names: List[str] = []
    rank_entries: List[dict] = []
    for gid in server_args.rank_gpu_id:
        entry = profile["gpus"].get(uuid_by_idx.get(gid, ""), None)
        if entry is None:
            lines.append(
                f"GPU {gid} missing from the profile -- keeping plain auto."
            )
            emit()
            return
        rank_entries.append(entry)
        rank_scores_bw.append(entry["membw_gbs"])
        gemv = entry.get("membw_gemv_gbs")
        if gemv is None:
            rank_scores_gemv = None
        elif rank_scores_gemv is not None:
            rank_scores_gemv.append(float(gemv))
        rank_names.append(entry["name"])

    # Compute score per rank, in the checkpoint's OWN weight format. The
    # prefill objective is a compute RATIO between ranks, and the ratio the
    # dense bf16 probe reports is not the ratio a quantized checkpoint runs at
    # -- on the reference rig, 3.79 in bf16 against 8.64 for the same two
    # cards on the fp8 checkpoint they actually serve (#296).
    # A MIXED_PRECISION checkpoint runs two formats on the same card, so the
    # score carries a family axis as well (see the "Compute families" block).
    # ``family_fmts`` is empty for every single-scheme checkpoint and the
    # scalar below is then the only vector in play.
    quant_fmt, quant_desc, family_fmts = checkpoint_compute_format_families(
        server_args.model_path
    )
    gemm = rank_gemm_family_scores(rank_entries, quant_fmt, family_fmts)
    rank_scores_gemm = gemm.scalar
    rank_gemm_lanes = gemm.scalar_labels
    lines.append(f"checkpoint weight format: {quant_desc}")
    for w in gemm.warnings:
        logger.warning("auto-performance: %s", w)
        lines.append("WARNING: " + w)
    for r in range(server_args.tp_size):
        gemv_txt = (
            f", decode GEMV {rank_scores_gemv[r]:.0f} GB/s"
            if rank_scores_gemv is not None
            else ""
        )
        lines.append(
            f"rank {r} -> GPU {server_args.rank_gpu_id[r]} ({rank_names[r]}): "
            f"GEMM {rank_scores_gemm[r]:.1f} TFLOPS [{rank_gemm_lanes[r]}], "
            f"membw {rank_scores_bw[r]:.0f} GB/s{gemv_txt}"
        )
    if gemm.mixed:
        lines.append(
            "MIXED-PRECISION checkpoint: the compute score is resolved per "
            "family, because two families run different lanes on the SAME "
            "card here."
        )
        for family in GEMM_FAMILIES:
            if family not in gemm.families:
                continue
            per_rank = ", ".join(
                f"r{r} {gemm.families[family][r]:.1f} "
                f"[{gemm.family_labels[family][r]}]"
                for r in range(server_args.tp_size)
            )
            lines.append(
                f"  family {family} ({gemm.family_formats[family]}): {per_rank}"
            )
    links = profile.get("links") or {}
    pair_bws = [v["p2p_gbs"] for k, v in links.items() if k != "__group__"]
    # No measured wire means the collective term is not priced and the
    # objective ranks on compute alone (#359). The term is split-invariant, so
    # the ordering is unchanged by omitting it; what used to happen instead was
    # an assumed 8.0 GB/s that made the reported margins depend on a number
    # nobody measured.
    min_link = min(pair_bws) if pair_bws else None
    if pair_bws:
        lines.append(
            "link matrix: "
            + ", ".join(
                f"{k.split('|')[0][-8:]}<->{k.split('|')[1][-8:]} "
                f"{v['p2p_gbs']:.1f} GB/s"
                for k, v in links.items()
                if k != "__group__"
            )
        )
    else:
        lines.append(
            "link matrix: not measured — the prefill objective prices compute "
            "only and its reported margin excludes the per-layer all-reduce"
        )

    # Measured KV-budget registry (previous boot of this exact config): when
    # complete, the capacity model runs on MEASURED residency posts instead
    # of the static heuristics, and the optimizer switches to the capacity
    # objective below. Consumed under the cross-algorithm gate ONLY — every
    # other configuration plans byte-identically with or without a registry.
    registry = (
        load_measured_registry(server_args)
        if getattr(server_args, "speculative_cross_algorithm", False)
        else None
    )
    model = PerfCostModel(
        PlanInputs.from_server_args(server_args),
        base_plan,
        budgets,
        measured=(registry or {}).get("components"),
        measured_mlp_vector=(registry or {}).get("mlp_vector"),
    )
    # The enc/both objective's only lever is moving MLP units, so it is scored
    # on the MLP family's own compute rate -- the routed-expert rate on a MoE
    # checkpoint, where that family carries the mass the vector moves. On every
    # single-scheme checkpoint neither family diverges, ``resolve`` returns the
    # scalar, and the objective sees the identical vector it saw before.
    enc_families = (
        (GEMM_FAMILY_MOE, GEMM_FAMILY_MLP)
        if model.num_experts > 0
        else (GEMM_FAMILY_MLP,)
    )
    base_enc_scores, enc_score_source = gemm.resolve(*enc_families)
    if gemm.mixed:
        lines.append(
            "enc/both objective scored on the "
            + (
                f"'{enc_score_source}' family lane"
                if enc_score_source != "scalar"
                else "checkpoint-wide scalar lane (no family datum for "
                + "/".join(enc_families)
                + ")"
            )
            + f": {[round(x, 1) for x in base_enc_scores]} TFLOPS."
        )
    # #287 per-family prefill arithmetic -- the piece #324 deliberately
    # deferred to keep its own merge byte-identical. On a MIXED checkpoint
    # the prefill time model divides each weight family's params by the rate
    # of the lane THAT family runs per rank (sum(p/r)); the scalar path sums
    # the params first and divides by one resolved rate (sum(p)/r), which
    # mis-times every rank whose families straddle lanes. ``None`` on every
    # single-scheme checkpoint: the time model then executes the pre-#287
    # float operations bit-for-bit.
    base_family_tflops = family_prefill_tflops(model, gemm)
    if base_family_tflops is not None:
        lines.append(
            "prefill time model: per-family sum(p/r) arithmetic ACTIVE "
            "(mixed checkpoint; diverging families: "
            + ", ".join(sorted(gemm.families))
            + ")"
        )
    # Which bandwidth the decode roofline divides by, stated once. The GEMV
    # rate is the informative one; every fall back to the streaming peak is
    # reported with its reason, because the two carry different exponents and
    # a silent swap between them is not a detail (see decode_bw_basis).
    _bw_rates, _bw_beta, _bw_basis = model.decode_bw_basis(
        rank_scores_bw, rank_scores_gemv
    )
    _bw_eff = model.effective_decode_bw(rank_scores_bw, rank_scores_gemv)
    _bw_eff_total = sum(_bw_eff) or 1.0
    lines.append(
        f"decode roofline divisor: {_bw_basis} "
        f"{[round(x, 1) for x in _bw_rates]} GB/s, residual exponent "
        f"{_bw_beta:g} -> effective bandwidth shares "
        + ", ".join(
            f"rank {r} {x / _bw_eff_total * 100:.1f}%"
            for r, x in enumerate(_bw_eff)
        )
    )

    # #434: the borrowed case is the one worth printing. These four scalars
    # were fitted on the machine this fork was developed on; on any other
    # machine they are a hypothesis the plan is priced with, and reporting
    # only the OVERRIDDEN case let silence stand for "reference fit".
    borrowed = model.calibration.borrowed_fields()
    if borrowed:
        lines.append(
            "calibration BORROWED (not measured here): "
            + ", ".join(borrowed)
            + " -- scalars fitted on the fork's development rig against one "
            "checkpoint family and one prefill mode, applied to this plan "
            "because nothing has refitted them for this machine. They set "
            "the decode roofline's residual exponent and the split-invariant "
            "time fractions, i.e. HOW MUCH a candidate is predicted to gain, "
            "not which cards are fast (that comes from the profile). Refit "
            "via the matching SGLANG_PERF_* env vars; each constant's "
            "definition in uneven_perf.py carries its campaign and recipe."
        )
    overridden = model.calibration.overridden_fields()
    if overridden:
        lines.append(
            "calibration OVERRIDDEN via SGLANG_PERF_* (refit seam): "
            + ", ".join(
                f"{name}={getattr(model.calibration, name):g}"
                for name in overridden
            )
            + " -- these replace the reference-rig fit for this plan."
        )

    base_pred = model.predict_capacity(base_plan)
    floor = (1.0 - loose / 100.0) * base_pred["ctx"]
    lines.append(
        f"VRAM-auto reference: predicted per-rank capacity "
        f"{[int(x) for x in base_pred['p']]} tokens, predicted max context "
        f"~{int(base_pred['ctx'])} (converged weighted-DCP optimum; "
        f"estimate), materialized MLP units "
        f"{model.mlp_unit_partition(base_plan)}"
    )
    lines.append(
        f"context floor (--rank-perf-loose-ctx-percent {loose:g}): "
        f"candidates must predict >= {int(floor)} tokens "
        f"({100 - loose:g}% of the VRAM-auto prediction)"
    )

    # --- fundability (#265) ------------------------------------------------
    # The floor above is a CONTEXT judgement. It is not the only way a
    # candidate can be wrong, and #264 measured the other way: 6,1,1 at the
    # runbook reserve does not boot at all -- it OOMs in the first real
    # prefill -- while the ladder printed "REJECTED by floor" for it, a
    # verdict whose documented remedy (raise --rank-perf-loose-ctx-percent)
    # trades context for an OOM instead of for speed.
    #
    # The mechanism is visible in the plan's own numbers. The KV pool is not
    # sized to a rank's capacity but to the token vector, scaled by the
    # TIGHTEST rank, so a rank that is not the tightest keeps its unused
    # capacity as free VRAM. Concentration moves the tight rank ONTO the card
    # being concentrated and spends exactly that slack: on the #264 rig the
    # 5090 goes from 4589 MiB of unused-capacity slack at the base to 0 at
    # 6,1,1, and its whole remaining headroom is then the pinned reserve
    # (3000 MiB) against a derived demand of 4160 MiB. Raising the reserve to
    # 4500 restores the margin, and that is the boot that ran.
    #
    # The check therefore asks one question per rank: does this candidate push
    # the rank from "residual free VRAM at or above the derived reserve
    # demand" to "below it"? Relative to the base on purpose -- a rank whose
    # reserve is ALREADY under-pinned (the 3080s here, at 2700 against 4160)
    # is the operator's pre-existing choice, already warned about at
    # resolution time, and not something a candidate caused.
    _fund_totals: Optional[List[int]] = None
    _fund_counts: Optional[List[int]] = None
    _fund_demand: Optional[List[int]] = None
    _fund_token_vec: Optional[List[int]] = None
    _fund_base: Optional[List[float]] = None
    #: True when the KV token vector the BOOT will run is MATCHED to the
    #: candidate being priced instead of fixed across candidates. Selects the
    #: gate's basis; see the comment at its assignment below.
    _fund_matched = False
    _fund_corridor = planner_corridor_mib()
    demand_by_gpu = getattr(server_args, "_derived_rank_auto_reserve_per_gpu", None)
    if demand_by_gpu:
        from collections import Counter

        from sglang.srt.distributed.utils import partition_units

        counts = Counter(server_args.rank_gpu_id)
        try:
            _fund_totals = [
                int(profile["gpus"][uuid_by_idx[g]]["total_mib"])
                for g in server_args.rank_gpu_id
            ]
            _fund_counts = [counts[g] for g in server_args.rank_gpu_id]
            _fund_demand = [demand_by_gpu[g] for g in server_args.rank_gpu_id]
        except (KeyError, TypeError, ValueError):
            _fund_totals = _fund_counts = _fund_demand = None
    if _fund_demand is not None:
        # The vector the BOOT will use, which is not always the converged
        # capacity optimum predict_capacity reports: 'coupled' (the default)
        # follows the base plan, a pinned vector follows itself, and the
        # derived modes install the capacity optimum after profiling.
        # #437: WHICH vector the boot runs decides which of two bases the
        # check can use, and the two are not interchangeable.
        #
        # A FIXED vector (the plain coupled default, or an explicit
        # --rank-kv-ratio pin) is the same for every candidate, so the
        # residual's second term -- ``(P_r - unit * v_r) * kv_cell``, capacity
        # the vector does not use -- really does move with the candidate.
        # Concentration relocates the tight rank and spends exactly that
        # slack, which is what #264 walked into, so pricing a candidate
        # RELATIVE to the base plan is both meaningful and correctly narrow:
        # a rank the operator already under-pinned is not the candidate's
        # fault.
        #
        # A MATCHED vector has no unused capacity by construction: the token
        # vector IS the capacity partition, so the slack term is ~0 on every
        # rank, for every candidate, including the base. Relative pricing then
        # compares two identical numbers and can never fire -- measured on the
        # #264/#354 fixture, `--rank-kv-ratio capacity` accepted 16,1,1 at
        # reserve 3000,2700,2700, the exact configuration #264 booted into an
        # OOM (41.69 MiB free on GPU 0), while the fixed-vector pricing
        # reported the same candidate UNBOOTABLE. The matched basis therefore
        # prices the reserve slack EXPLICITLY and ABSOLUTELY instead
        # (_unfundable_reason below), on every card rather than only on the
        # binding one: #435 moved the risk off rank 0 and onto the ranks that
        # used to keep the slack (#433 measured 7045 / 7109 MiB idle on ranks
        # 1 and 2 -- matching spends it; NOTE_433, "Corridor gate, both
        # sub-arms").
        kv_ratio = getattr(server_args, "rank_kv_ratio", "coupled")
        if isinstance(kv_ratio, list):
            _fund_token_vec = [max(int(v), 1) for v in kv_ratio]
        elif server_args.uneven_kv_derived_mode():
            # 'capacity'/'speed': the converged optimum, per candidate.
            _fund_token_vec = None
            _fund_matched = True
        elif _phase_solve_owns_kv_ratio(server_args, model, tune):
            # #435: the phase solve seeds the chosen candidate's own matched
            # vector into the boot, so the boot no longer runs the base-plan
            # split this gate used to price it against.
            _fund_token_vec = None
            _fund_matched = True
        else:
            _fund_token_vec = partition_units(_PREDICT_TOKEN_UNITS, base_plan)

    def _residual(cand: Sequence[int]) -> Optional[List[float]]:
        """Per-rank residual free MiB for a candidate, or None when the
        inputs for the check are not available."""
        if _fund_demand is None or _fund_totals is None or _fund_counts is None:
            return None
        vec = _fund_token_vec
        if vec is None:
            vec = model.predict_capacity(list(cand))["token_vector"]
            if not vec:
                return None
        return model.residual_free_mib(
            list(cand), _fund_totals, _fund_counts, vec
        )

    def _unfundable_reason(res: Optional[List[float]]) -> Optional[str]:
        """Verdict text for the first rank that cannot fund its own boot, or
        None. Two bases, per ``_fund_matched`` above.

        FIXED vector (relative): the first rank the candidate pushes below
        the derived reserve demand that the BASE plan still clears.

        MATCHED vector (absolute): the first rank whose residual does not
        cover the derived reserve demand at all. There is nothing to compare
        against here -- the base spends the same slack -- so the demand is
        the whole test, on every rank rather than on the binding one.
        """
        if res is None or _fund_demand is None:
            return None
        if not _fund_matched and _fund_base is None:
            return None
        for r in range(server_args.tp_size):
            if _fund_matched:
                unfunded = res[r] < _fund_demand[r]
                context = (
                    "matched KV vector: the boot spends this rank's unused "
                    "capacity, so the reserve is its whole remaining headroom"
                )
            else:
                unfunded = res[r] < _fund_demand[r] <= _fund_base[r]
                context = (
                    f"the base plan leaves it {int(_fund_base[r])} MiB"
                )
            if unfunded:
                # Deliberately worded away from the floor: an unbootable
                # candidate has no context to trade, so pointing at
                # --rank-perf-loose-ctx-percent would buy an OOM (#264).
                return (
                    f"UNBOOTABLE (rank {r} residual free {int(res[r])} MiB < "
                    f"derived reserve demand {_fund_demand[r]} MiB; "
                    f"{context}). Not "
                    "acceptable at any --rank-perf-loose-ctx-percent -- raise "
                    "--rank-auto-reserve-mib on that GPU instead"
                )
        return None

    def _corridor_note(res: Optional[List[float]]) -> Optional[str]:
        """Advisory text for ranks that clear the demand but land inside
        #330's absolutely-free corridor, or None.

        Reported, never binding. The demand is a MODEL of the non-budget
        posts and the corridor is what a boot must leave beyond them; a
        candidate sitting between the two is one the operator may still want
        (#354's 16,1,1 booted and served at 87 MiB free on the 5090), but it
        is not one anybody should choose without being told. #424 made that
        call by hand off a post-boot number; this states it before the boot.
        """
        if res is None or _fund_demand is None or _fund_corridor <= 0:
            return None
        tight = [
            (r, int(res[r] - _fund_demand[r]))
            for r in range(server_args.tp_size)
            if 0 <= res[r] - _fund_demand[r] < _fund_corridor
        ]
        if not tight:
            return None
        return (
            "CORRIDOR-TIGHT ("
            + ", ".join(
                f"rank {r} predicted free-after-boot {m} MiB" for r, m in tight
            )
            + f" < the {_fund_corridor} MiB absolutely-free corridor of #330)"
        )

    _fund_base = _residual(base_plan)
    if _fund_base is not None and _fund_demand is not None:
        if _fund_matched:
            lines.append(
                "fundability basis: MATCHED KV token vector -- the boot runs "
                "each candidate's own capacity partition, so no rank keeps "
                "unused capacity as free VRAM and every candidate is priced "
                "ABSOLUTELY, on all ranks: residual free VRAM must cover the "
                f"derived reserve demand {_fund_demand} MiB. At the VRAM-auto "
                f"split that residual is {[int(x) for x in _fund_base]} MiB "
                "per rank. Pricing this relative to the base instead would "
                "compare two identical numbers and accept everything (#437; "
                "#264 measured the OOM it accepted)."
            )
            base_unfundable = _unfundable_reason(_fund_base)
            if base_unfundable:
                lines.append(
                    "fundability WARNING: the VRAM-auto split ITSELF does not "
                    f"clear the demand on this reserve -- {base_unfundable}. "
                    "No MLP candidate can repair that; every candidate is "
                    "reported UNBOOTABLE and the plain split this falls back "
                    "to carries the same risk. Raise --rank-auto-reserve-mib "
                    "on the named GPU(s) or lower --context-length."
                )
        else:
            lines.append(
                "fundability reference: residual free VRAM at the VRAM-auto "
                f"split {[int(x) for x in _fund_base]} MiB per rank (reserve the "
                "budget never claims + capacity the KV token vector does not "
                f"use), against the derived reserve demand {_fund_demand} MiB. A "
                "candidate that pushes a rank from above that demand to below it "
                "is reported UNBOOTABLE and is never accepted, whatever "
                "--rank-perf-loose-ctx-percent says."
            )
        base_corridor = _corridor_note(_fund_base)
        if base_corridor:
            lines.append(f"fundability corridor at the VRAM-auto split: {base_corridor}")

    # Tuning target (per M22: decode is flat across representable splits;
    # prefill/aggregate is the lever, and 'both' rides the same lever).
    chosen: Optional[Tuple[int, ...]] = None
    # Capacity-directed planning (T156 option 1): under the cross gate the
    # solo host's non-weight posts (global-context DFLASH draft cell,
    # measured graph/workspace residency) make the GLOBAL capacity a steep
    # function of the host's weight share — the speed objective below, blind
    # to that, parked the MLP mass on the host and left the shadow cards
    # ~5 GiB idle while C was host-bound. With a complete measured registry
    # the capacity model is trustworthy in ABSOLUTE terms, so the objective
    # flips: maximize predicted context, tie-break by prefill speed among
    # near-optimal (>=99%) candidates. The decode-knee guard is LOGGED but
    # not binding here — trading decode round time for capacity is the
    # user's explicit decision for this mode; the measured delta is reported
    # after boot. Gated on cross_active + registry + capacity KV mode, so
    # every other configuration keeps the speed objective byte-identically.
    capacity_directed = bool(
        model.cross_active
        and model.measured is not None
        and server_args.uneven_kv_capacity_mode()
    )
    if capacity_directed:
        import itertools

        lines.append(
            "capacity-directed planning ACTIVE (cross-algorithm gate + "
            "complete measured registry): objective = predicted max context "
            "on the measured capacity model; prefill speed breaks ties "
            "within 1%; decode-knee guard is advisory (logged only)."
        )
        enc_scores = list(base_enc_scores)
        # Candidate space: exhaustive small-integer vectors (the capacity
        # optimum usually DRAINS the solo host, a direction the
        # score-proportional ladders never propose). 6^tp stays trivial for
        # tp<=4; larger groups fall back to the ladders + the even vector.
        if 6 ** model.tp_size <= 1296 * 6:
            cand_set = {
                _gcd_reduce(v)
                for v in itertools.product(
                    range(1, 7), repeat=model.tp_size
                )
            }
        else:
            cand_set = set(_mlp_candidates(model, enc_scores, base_plan))
            cand_set.add(tuple([1] * model.tp_size))
        cand_set.add(_gcd_reduce(base_plan))

        scored = []
        unfundable_seen = 0
        for cand in sorted(cand_set):
            pred = model.predict_capacity(list(cand))
            if not pred["feasible"]:
                continue
            # #437: this branch used to filter on ``feasible`` alone, i.e. on
            # "the capacity model returns a positive per-rank capacity", and
            # never consulted the fundability gate at all -- so the one
            # objective that maximizes context was also the one with no check
            # that the winning context can be allocated. Fundability is a
            # correctness gate in every other objective and it is one here.
            if _unfundable_reason(_residual(cand)) is not None:
                unfundable_seen += 1
                continue
            t_cand = model.prefill_time_model(
                list(cand), enc_scores, min_link, base_family_tflops
            )
            scored.append((cand, pred, t_cand))
        if unfundable_seen:
            lines.append(
                f"capacity objective: {unfundable_seen} candidate(s) dropped "
                "as UNBOOTABLE by the fundability gate before scoring."
            )
        if scored:
            best_ctx = max(p["ctx"] for _, p, _ in scored)
            near = [x for x in scored if x[1]["ctx"] >= 0.99 * best_ctx]
            # HYSTERESIS: prefer the INCUMBENT (the vector the registry was
            # measured under) whenever it is itself near-optimal. Without
            # this the choice oscillates around the host's feasibility edge
            # (measured 2026-07-22: 2,1,1 <-> 3,1,1 flip every boot, because
            # the bias anchor is exact only AT the measured vector), and
            # since budget corrections are vector-specific, every flip
            # resets them and the fill never converges. Switching still
            # happens the moment the incumbent falls out of the 1% window
            # (a real capacity gain).
            incumbent = None
            if model.measured_mlp_vector is not None:
                inc_vec = _gcd_reduce(model.measured_mlp_vector)
                for x in near:
                    if x[0] == inc_vec:
                        incumbent = x
                        break
            if incumbent is not None:
                cand, pred, t_cand = incumbent
                lines.append(
                    f"capacity objective: incumbent vector "
                    f"{','.join(map(str, cand))} (measured registry vector) "
                    "is within 1% of the optimum -- kept for correction "
                    "convergence (hysteresis)."
                )
            else:
                cand, pred, t_cand = min(near, key=lambda x: x[2])
            # Decode round-time delta vs base. Built on EFFECTIVE bandwidth:
            # the same expression over PROBED PEAK bandwidth reported -22 %
            # for a vector that measured +16.5 % (task #216).
            dec_delta = model.decode_cost_percent(
                list(cand), rank_scores_bw, rank_scores_gemv
            )
            knee_ok, knee_reason = model.decode_knee_detail(
                list(cand), rank_scores_bw, rank_scores_gemv
            )
            t_base = model.prefill_time_model(
                base_plan, enc_scores, min_link, base_family_tflops
            )
            for c2, p2, t2 in sorted(scored, key=lambda x: -x[1]["ctx"])[:6]:
                note2 = _corridor_note(_residual(c2))
                lines.append(
                    f"capacity candidate {','.join(map(str, c2))}: predicted "
                    f"ctx ~{int(p2['ctx'])}, per-rank weights "
                    f"{[round(w, 2) for w in p2['weights_gib']]} GiB, "
                    f"prefill {t_base / t2 - 1.0:+.1%} vs base"
                    + (f" -- {note2}" if note2 else "")
                )
            lines.append(
                f"capacity objective: best predicted ctx ~{int(best_ctx)}; "
                f"chosen {','.join(map(str, cand))} (within 1%, fastest "
                f"prefill); predicted decode round-time delta {dec_delta:+.1f}% "
                f"(streamed-bytes/membw proxy; ADVISORY decode-knee: "
                f"{'ok' if knee_ok else knee_reason})"
            )
            if _gcd_reduce(cand) != _gcd_reduce(base_plan):
                chosen = cand
            else:
                lines.append(
                    "capacity optimum IS the VRAM-auto base split -- no MLP "
                    "override needed."
                )
        else:
            lines.append(
                "capacity-directed planning: no feasible AND fundable "
                "candidate on the measured model -- keeping plain auto "
                "(registry stale? budgets shrank? reserve too low -- see the "
                "fundability lines above)."
            )
    else:
        # The decode-knee guard REJECTS in the single-vector modes and only
        # REPORTS in the phase-optimal modes (#357). It is not being softened:
        # its number was right. At the #354 operating point it predicted a
        # +24.7% decode step for 16,1,1 and the boot measured +20.2% (bs=8) to
        # +25.0% (bs=1) -- but that cost belongs to the DECODE phase, which in
        # this recipe runs the companion VRAM-auto vector. Vetoing the prefill
        # vector on it rejected a vector for a price the recipe never pays:
        # the same boot measured +22.6% prefill at s=1 against a +22.9%
        # prediction. Fundability and the context floor stay binding in every
        # mode -- those are about whether the boot happens at all.
        knee_binding = tune not in _PHASE_TUNES
        if tune == "both":
            lines.append(
                "tune=both: prefill and concurrent throughput ride the same "
                "MLP-concentration lever (M22: +10% prefill / +7% conc-8 "
                "for the C6-class vector), so 'both' optimizes the same "
                "objective as 'enc'."
            )
        elif tune == "dec":
            # #434: this target used to print M22's reference-rig finding
            # ("decode is FLAT across all representable splits") and return
            # the base split WITHOUT running the optimizer. That is a
            # measurement from one rig (5090 + 2x 3080) asserted as a
            # property of every rig. It is not one: the base plan is
            # proportional to VRAM BUDGET and the decode round time is set by
            # streamed weight bytes over EFFECTIVE BANDWIDTH, so the two
            # coincide only while a card's VRAM share equals its bandwidth
            # share. On the reference rig they nearly do, which is what M22
            # measured; on a rig whose capacity and bandwidth rankings
            # disagree -- a large slow card next to a small fast one, an
            # NVLink island, an 8x-equal box with one throttled card -- the
            # decode lever is real and the old branch could never find it.
            # The objective below is solved from THIS rig's own profile
            # scores; where decode really is flat, no candidate beats the
            # base and the refusal says so by name.
            lines.append(
                "tune=dec: objective = minimize the lockstep bs=1 DECODE "
                "round time (streamed weight bytes / effective decode "
                "bandwidth, per rank), solved from this rig's own measured "
                "profile. The weight split is only one of the two decode "
                "levers -- the other is the KV-TOKEN split, which this "
                "target selects separately via --rank-kv-ratio speed. On a "
                "rig whose VRAM shares already match its bandwidth shares "
                "the weight lever is flat (M22 measured exactly that on the "
                "reference rig) and the refusal below names it; that is a "
                "result of the solve here, not an assumption."
            )
        elif tune in _PHASE_TUNES:
            lines.append(
                f"tune={tune}: PHASE-OPTIMAL recipe (#354/#357). One weight "
                "vector per phase instead of one for the whole server: the "
                "prefill arm is the concentrated MLP vector solved below, the "
                "decode arm is the plain VRAM-auto split (measured decode "
                "optimum, #265/#354). The decode-knee guard is ADVISORY here "
                "and its number is printed for every candidate -- the vector "
                "it warns about is not the vector that decodes. Fundability "
                "and the context floor still REJECT."
            )
        else:
            # tune=enc used to fall into the 'both' branch without a word of
            # its own, which made "enc did nothing" indistinguishable from
            # "enc had nothing to do" (#264 bug 1). enc has exactly ONE
            # lever -- moving MLP units onto the compute-strong rank -- so
            # when every concentrated candidate is rejected, enc has no lever
            # at this operating point and says so below instead of quietly
            # returning the base split as if it had optimized something.
            lines.append(
                "tune=enc: objective = minimize the lockstep PREFILL time "
                "model. The only lever is MLP concentration onto the "
                "compute-strong rank(s); the attention/GDN/KV splits are not "
                "touched (M22: decode is flat across them). If every "
                "concentrated candidate is rejected below, enc has no lever "
                "here and the refusal is reported explicitly."
            )
        # enc/both objective: minimize the lockstep prefill time model.
        enc_scores = list(base_enc_scores)
        enc_family_tflops = base_family_tflops
        # Per-rank link penalty: a rank behind a narrow link attracts fewer
        # units (folded softly; the AR term already carries the group cost).
        if pair_bws and len(set(server_args.rank_gpu_id)) == server_args.tp_size:
            per_rank_link = []
            for gid in server_args.rank_gpu_id:
                u = uuid_by_idx[gid]
                bws = [
                    v["p2p_gbs"]
                    for k, v in links.items()
                    if k != "__group__" and u in k
                ]
                per_rank_link.append(max(bws) if bws else min_link)
            best = max(per_rank_link)
            enc_scores = [
                s * (l / best) ** _PREDICT_LINK_ALPHA
                for s, l in zip(enc_scores, per_rank_link)
            ]
            if enc_family_tflops is not None:
                # The penalty is a per-RANK factor (the link is a property of
                # the card, not of the format), so it scales every family's
                # vector identically.
                enc_family_tflops = {
                    name: [
                        r * (l / best) ** _PREDICT_LINK_ALPHA
                        for r, l in zip(rates, per_rank_link)
                    ]
                    for name, rates in enc_family_tflops.items()
                }
        t_base = model.prefill_time_model(
            base_plan, enc_scores, min_link, enc_family_tflops
        )
        # #434: which quantity the ladder MAXIMIZES the gain of. Every target
        # but 'dec' scores the prefill time model; 'dec' scores the bs=1
        # decode round time. Both are built from THIS rig's own profile
        # entries -- GEMM lanes for prefill, effective decode bandwidth for
        # decode -- so neither carries a reference-rig verdict into the
        # answer.
        decode_objective = tune == "dec"
        dec_scores = model.effective_decode_bw(rank_scores_bw, rank_scores_gemv)
        objective_metric = "decode" if decode_objective else "prefill"
        candidates = _mlp_candidates(model, enc_scores, base_plan)
        if decode_objective:
            # Concentration toward the BANDWIDTH-strong rank is the decode
            # ladder; the compute ladder is kept in the set because on a rig
            # where the two rankings agree it names the same vectors, and
            # where they disagree an operator reading the log gets to see
            # both priced. Union, deduplicated, base excluded.
            seen_cands = {tuple(c) for c in candidates}
            for cand in _mlp_candidates(model, list(dec_scores), base_plan):
                if tuple(cand) not in seen_cands:
                    seen_cands.add(tuple(cand))
                    candidates.append(cand)
        best_gain = 0.0
        # #350 phase 4: the boot's own objective. Resolved and validated
        # HERE, at parse time, so an energy request that cannot be priced
        # fails the boot with a named reason instead of quietly booting the
        # throughput vector under an energy flag (the fork's validate-early
        # rule; the same refusal solve() makes in the planner).
        energy_model, best_joules, energy_scores = None, math.inf, {}
        if _objective_is_energy(server_args):
            from sglang.srt.planner.objective import energy_per_work

            energy_model = _boot_energy_model(server_args, model, lines)
        results = []
        for cand in candidates:
            pred = model.predict_capacity(list(cand))
            t_cand = model.prefill_time_model(
                list(cand), enc_scores, min_link, enc_family_tflops
            )
            gain = t_base / t_cand - 1.0
            knee_ok, knee_reason = model.decode_knee_detail(
                list(cand), rank_scores_bw, rank_scores_gemv
            )
            if decode_objective:
                # Speedup fraction of the whole decode STEP, so the number is
                # comparable with the prefill gain above and with the
                # per-candidate decode line printed further down (which is
                # the same quantity expressed as a cost).
                step_ratio = 1.0 + model.decode_cost_percent(
                    list(cand), rank_scores_bw, rank_scores_gemv
                ) / 100.0
                gain = (1.0 / step_ratio - 1.0) if step_ratio > 0 else -1.0
            # Tolerance, not slop (#265): re-partitioning the MLP family
            # CONSERVES total weight bytes, so sum_r P_r -- and with it the
            # predicted context whenever the sum is the binding term -- is
            # the same number for every candidate in exact arithmetic. Only
            # the summation ORDER differs, and a bare >= then rejects a
            # capacity-NEUTRAL candidate on the last bits: the #264 log shows
            # six candidates "REJECTED by floor" at a printed ctx identical to
            # the floor's own 492416. A relative tolerance is the correct
            # comparison for a quantity that is mathematically equal.
            floor_ok = pred["feasible"] and (
                pred["ctx"] >= floor
                or math.isclose(pred["ctx"], floor, rel_tol=1e-9)
            )
            res = _residual(cand)
            unfundable = _unfundable_reason(res)
            results.append(
                (cand, pred, gain, floor_ok, knee_ok, knee_reason, res, unfundable)
            )
            admissible = (
                floor_ok and (knee_ok or not knee_binding) and unfundable is None
            )
            if energy_model is not None:
                # #350 phase 4: the ENERGY objective changes only WHICH
                # admissible candidate wins, never which are admissible --
                # the context floor, the decode-knee guard and fundability
                # are correctness gates and a joule does not buy past them.
                # Price = lockstep time x summed board power over the SAME
                # per-rank times the throughput model maxes over -- prefill
                # GEMM times for every target but 'dec', decode weight-stream
                # times for 'dec' (#434). Pricing a decode objective on
                # prefill times would rank the candidates by a round the
                # target does not optimize.
                if admissible:
                    j = energy_per_work(
                        model.per_rank_decode_times(
                            list(cand), rank_scores_bw, rank_scores_gemv
                        )
                        if decode_objective
                        else model.per_rank_prefill_compute_times(
                            list(cand), enc_scores, enc_family_tflops
                        ),
                        energy_model,
                    )
                    energy_scores[tuple(cand)] = j
                    if j < best_joules - 1e-12:
                        best_joules = j
                        chosen = cand
            elif admissible and gain > best_gain + 1e-9:
                best_gain = gain
                chosen = cand
        for (
            cand,
            pred,
            gain,
            floor_ok,
            knee_ok,
            knee_reason,
            res,
            unfundable,
        ) in sorted(results, key=lambda x: -x[2])[:6]:
            if not pred["feasible"]:
                verdict = "INFEASIBLE"
            elif unfundable is not None:
                # Deliberately ahead of the floor verdict: an unbootable
                # candidate has no context to trade, so naming the floor here
                # would send the reader to --rank-perf-loose-ctx-percent, and
                # that knob buys an OOM (#264).
                verdict = unfundable
            elif not floor_ok:
                verdict = "REJECTED by floor"
            elif not knee_ok and knee_binding:
                verdict = f"REJECTED by decode-knee guard -- {knee_reason}"
            elif not knee_ok:
                verdict = (
                    "floor OK, fundable; decode-knee ADVISORY (not binding in "
                    f"a phase-optimal arm) -- {knee_reason}"
                )
            else:
                verdict = "floor OK, knee OK, fundable"
            if unfundable is None:
                # #437: a candidate can clear the demand and still leave the
                # card with no absolutely-free room. That is a decision, not a
                # rejection, so it rides along with the verdict instead of
                # replacing it.
                note = _corridor_note(res)
                if note:
                    verdict = f"{verdict}; {note}"
            # The decode cost is stated for EVERY candidate, including the
            # ones the guard accepts. A silent pass is how a +16.5 % decode
            # regression got proposed as a default (task #216): the guard's
            # verdict is a model, the number next to it is what lets a reader
            # disagree with the model.
            dec_pct = model.decode_cost_percent(
                list(cand), rank_scores_bw, rank_scores_gemv
            )
            lines.append(
                f"candidate MLP vector {','.join(map(str, cand))}: "
                f"predicted ctx ~{int(pred['ctx'])} ({verdict}), "
                f"predicted {objective_metric} gain {gain * 100:+.1f}%, "
                f"predicted decode step {dec_pct:+.1f}%"
                + (
                    f", residual free {[int(x) for x in res]} MiB"
                    if res is not None
                    else ""
                )
                + f" (units {model.mlp_unit_partition(list(cand)) if pred['feasible'] else 'n/a'})"
                + (
                    f", predicted {energy_scores[tuple(cand)]:.4g} J/token"
                    if tuple(cand) in energy_scores
                    else ""
                )
            )
        if chosen is None:
            lines.extend(
                _no_lever_lines(
                    tune, loose, results, knee_binding, objective_metric
                )
            )
        elif tune == _PHASE_DECODE:
            # The decode arm of the recipe IS the VRAM-auto split. The
            # concentrated vector is still solved and named, so one plan run
            # states both arms and the operator can drive the switch; it is
            # not installed here.
            lines.append(
                "phase-optimal DECODE arm: keeping the plain VRAM-auto split "
                "(the measured decode optimum). The companion PREFILL vector "
                f"for this rig and checkpoint is {','.join(map(str, chosen))} "
                f"(predicted prefill gain {best_gain * 100:+.1f}%); launch "
                "with --rank-perf-tune phase-prefill to serve the prefill "
                "arm."
            )
            chosen = None

    if chosen is not None:
        pred = model.predict_capacity(list(chosen))
        server_args.rank_mlp_ratio = list(chosen)
        lines.append(
            f"CHOSEN MLP vector: {','.join(map(str, chosen))} "
            f"(materialized units {model.mlp_unit_partition(list(chosen))}; "
            f"predicted ctx ~{int(pred['ctx'])} >= floor {int(floor)}; "
            f"predicted per-rank capacity {[int(x) for x in pred['p']]}; "
            f"predicted decode step "
            f"{model.decode_cost_percent(list(chosen), rank_scores_bw, rank_scores_gemv):+.1f}% "
            f"vs the VRAM-auto split)"
        )
        lines.append(
            "floor check: predicted ctx of chosen vector "
            f"{int(pred['ctx'])} >= {int(floor)} "
            f"({100 - loose:g}% of VRAM-auto {int(base_pred['ctx'])}) -- OK"
        )
        lines.append(
            f"PIN HINT: skip probe+optimizer on later boots with "
            f"--rank-tp-ratio auto --rank-mlp-ratio "
            f"{','.join(map(str, chosen))}"
        )
        if tune == "dec":
            # #434: the weight lever for decode is small wherever a card's
            # VRAM share is close to its bandwidth share, and the model that
            # sizes it was calibrated against step times whose boot-to-boot
            # floor is of the same order as the deltas it now reports. State
            # that next to the number instead of letting a predicted percent
            # read as a measured one. No numeric gate is applied: a threshold
            # would be a constant fitted somewhere, which is the defect this
            # target was changed to remove.
            lines.append(
                "tune=dec: the decode weight lever is a PREDICTION from the "
                "bandwidth profile, not a measurement. Confirm it against a "
                "same-boot A-vs-A floor (identical draws, both vectors, one "
                "boot each) before treating the delta as real -- and note "
                "that the KV-TOKEN lever this target also selects "
                "(--rank-kv-ratio speed) is the larger of the two decode "
                "levers wherever context is deep."
            )
        if tune == _PHASE_PREFILL:
            # State the other arm of the recipe, and what switching costs
            # today: the MLP vector is a WEIGHT split, and no runtime actuator
            # moves weights -- #297 (/kv_reshard) moves KV tokens, #330
            # (--enable-vram-dial) moves the VRAM budget (#354, measured).
            lines.append(
                "phase-optimal PREFILL arm installed. The companion DECODE "
                "arm is the plain VRAM-auto split (launch the same command "
                "with --rank-perf-tune phase-decode). Switching arms needs a "
                "RESTART: the MLP vector is a weight split and no runtime "
                "actuator moves weights (#297 moves KV tokens, #330 moves the "
                "VRAM budget)."
            )
        # Seed the DCP token vector from the PREDICTED per-rank capacity
        # instead of letting resolve_cp_token_ratios fall back to its budget
        # estimate. Two placements need this, for two different reasons; both
        # are cases where the estimate ("tokens proportional to VRAM budget
        # minus a weight share") no longer describes the boot.
        #
        # Precedence is respected in both: SGLANG_UNEVEN_TOKEN_VECTOR and an
        # explicit --rank-kv-ratio pin still win, because this only fills in
        # the unpinned default.
        if pred.get("token_vector"):
            existing_kv = getattr(server_args, "rank_kv_ratio", None)
            if not isinstance(existing_kv, list):
                tok_vec = [int(v) for v in pred["token_vector"]]
                if len(tok_vec) == model.tp_size and all(v > 0 for v in tok_vec):
                    g = math.gcd(*tok_vec)
                    tok_vec = [v // g for v in tok_vec]
                    if len(set(tok_vec)) > 1 and model.solo_active:
                        # Solo placement: the estimate is wrong on the HOST by
                        # two large, unmodelled costs -- the whole unsharded
                        # draft and a draft KV pool sized to the GLOBAL
                        # context. The host therefore gets a token share it
                        # cannot fund, the global pool is
                        # min_r(P_r * S / ratio_r) -- so the host binds it --
                        # and the other cards sit half empty. The predicted
                        # vector already accounts for both (see
                        # _solo_rank_token_capacity).
                        #
                        # 'capacity' mode: keep the MODE STRING intact and park
                        # the prediction in the dedicated seed field. Writing
                        # the vector into rank_kv_ratio itself would turn the
                        # mode into an explicit PIN
                        # (uneven_kv_capacity_mode() -> False), which cancels
                        # the phase-2 measured install after profiling and
                        # leaves the boot stuck on this pre-boot prediction.
                        # For every other value ('coupled', unset) the previous
                        # behavior is kept byte-identical.
                        if server_args.uneven_kv_derived_mode():
                            server_args.rank_kv_capacity_seed = tok_vec
                            seeded_as = "phase-1 seed; the measured install after profiling still runs"
                        else:
                            server_args.rank_kv_ratio = tok_vec
                            seeded_as = "explicit vector"
                        lines.append(
                            "draft-solo: seeded the DCP token vector from the "
                            f"predicted per-rank capacity -> {','.join(map(str, tok_vec))} "
                            f"({seeded_as}; the budget-estimate fallback does not model the "
                            "solo host's draft weights + global draft KV pool, "
                            "which would leave the shadow ranks half empty)."
                        )
                    elif len(set(tok_vec)) > 1 and _phase_solve_owns_kv_ratio(
                        server_args, model, tune
                    ):
                        # PHASE SOLVE (#435): the estimate is wrong because the
                        # MLP vector just moved weight mass AWAY from the
                        # budget proportion the estimate assumes. Every number
                        # printed above -- the predicted per-rank capacity, the
                        # predicted ctx, and therefore the floor gate the
                        # candidate had to clear -- is computed with the
                        # capacity-MATCHED token vector
                        # (predict_capacity: ctx = min(sum P, 64 * min P), which
                        # is exactly the capacity of partition_units(64, P)).
                        # The boot resolved the budget split instead, so the
                        # gate passed on a context the runtime then did not
                        # deliver.
                        #
                        # Measured, #433 addendum: solved MLP vector 8,1,1 on
                        # INT8, matched token vector [12,26,26], booted vector
                        # [31,17,16] -- rank 0 asked to own 48 % of the tokens
                        # while holding 13 % of the capacity, and the global
                        # pool (min-reduced over ratio-normalised capacity)
                        # came out at 125 504 tokens against a predicted
                        # 358 693.
                        #
                        # Written to rank_kv_capacity_seed, never to
                        # rank_kv_ratio: the mode STRING carries side effects
                        # of its own (uneven_kv_flag_active() gates the
                        # dcp_size auto-set and the weighted owner rule, both
                        # resolved AFTER this call), and matching a token
                        # vector must not silently switch a parallelism mode
                        # on. In resolve_cp_token_ratios the seed sits below
                        # the env override and the explicit pin and above the
                        # budget estimate, which is exactly the precedence
                        # this needs.
                        server_args.rank_kv_capacity_seed = tok_vec
                        lines.append(
                            "coupled KV ratio MATCHED to the solve: seeded the "
                            "DCP token vector from the chosen candidate's "
                            "predicted per-rank capacity -> "
                            f"{','.join(map(str, tok_vec))} (the budget-split "
                            "estimate no longer matches the MLP vector, and "
                            "the floor gate above is evaluated with this "
                            "vector). Pass --rank-kv-ratio explicitly to "
                            "override; --rank-kv-ratio capacity additionally "
                            "re-installs the MEASURED optimum after profiling, "
                            "which this pre-boot prediction only approximates."
                        )
    else:
        lines.append(
            "CHOSEN: keep plain VRAM-auto split (no MLP vector override)."
        )

    # --rank-kv-ratio speed needs integer weights proportional to the per-rank
    # MEASURED memory bandwidth, and this is the only place the hardware
    # profile is read. Park them on server_args for the phase-2 install after
    # the post-weight-load profiling (model_runner_kv_cache_mixin.
    # _maybe_suggest_dcp_token_vector). Always parked, not only in speed mode:
    # it is a pure record of the profile, costs nothing, and keeps the mode
    # switchable without re-probing. Same integer normalisation as the vocab
    # hint -- both want "weights proportional to bandwidth", and reusing the
    # helper keeps a single definition of that.
    server_args.rank_kv_speed_weights = vocab_ratio_from_membw(rank_scores_bw)
    lines.append(
        "KV-SPEED WEIGHTS: per-rank memory-bandwidth proportion is "
        f"{','.join(map(str, server_args.rank_kv_speed_weights))} "
        "(consumed by --rank-kv-ratio speed / --rank-perf-tune dec; under "
        "DCP each rank runs attention over the tokens it owns, so at bs=1 "
        "the deep-context part of the decode step follows this vector)."
    )

    # Ratio-weighted vocab sharding hint (--rank-vocab-ratio, M20 BEIFANG 2):
    # a separate opt-in flag (never applied here); the membw-weighted vocab
    # split balances the per-rank lm_head read time (~+4% MTP decode class).
    if server_args.rank_vocab_ratio is None:
        vocab_vec = vocab_ratio_from_membw(rank_scores_bw)
        lines.append(
            "VOCAB HINT: the membw-weighted vocab shard for embed/lm_head "
            f"is --rank-vocab-ratio {','.join(map(str, vocab_vec))} "
            "(or 'auto'; separate opt-in flag, balances the per-rank "
            "lm_head read time -- helps MTP drafts and every sampling step)."
        )

    rec = _tp_drop_recommendation(server_args, profile, gpus, model)
    if rec:
        lines.append(rec)
    emit()


# ---------------------------------------------------------------------------
# Probe subprocess entry point.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="sglang auto-performance probe")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help=(
            "comma-separated probe groups to measure ("
            + ", ".join(_PROBE_GROUP_FIELDS)
            + "). Default: all of them plus the pairwise link matrix. A subset "
            "is the LAZY TOP-UP mode -- the existing profile at --out is the "
            "base, only the named groups are re-measured, and the link matrix "
            "is carried over rather than re-run."
        ),
    )
    args = parser.parse_args()
    if args.probe:
        out = args.out
        if out is None:
            gpus, driver = _nvml_gpu_inventory()
            out = profile_cache_path([g["uuid"] for g in gpus], driver)
        groups = None
        base = None
        if args.groups:
            groups = [g.strip() for g in args.groups.split(",") if g.strip()]
            unknown = [g for g in groups if g not in _PROBE_GROUP_FIELDS]
            if unknown:
                parser.error(
                    f"unknown probe group(s) {unknown}; known: "
                    f"{sorted(_PROBE_GROUP_FIELDS)}"
                )
            if not os.path.exists(out):
                parser.error(f"--groups needs an existing base profile at {out}")
            with open(out) as f:
                base = json.load(f)
        # Environment pre-check, at the probe's actual entry point, BEFORE
        # run_probe touches a single card or writes anything (task #310): the
        # 'lanes' group's fp8 Marlin lane needs real sgl_kernel scalar types,
        # and an interpreter without them must abort loud rather than have
        # its AttributeErrors recorded as card-level GEMM failures.
        effective_groups = groups if groups is not None else list(_PROBE_GROUP_FIELDS)
        if PROBE_GROUP_LANES in effective_groups:
            try:
                _check_lane_probe_environment()
            except ProbeEnvironmentError as ex:
                print(f"{PROBE_ENV_ERROR_PREFIX}{ex}", file=sys.stderr)
                sys.exit(1)
        prof = run_probe(out, groups=groups, base=base)
        print(json.dumps(prof, indent=1))
    else:
        parser.error("nothing to do (pass --probe)")
