# Task #337 -- card runsheet: TensorRT-RTX vs. our kernels, per part

Desk work is done and the engines already exist. This sheet is everything the
card operator needs.

**Total card time: about 16 minutes for both arms** (the fold arms ride in the
same rotation; 12 min without them). The engines were built on a
CPU with no GPU visible, so no build step is on the critical path.

## 0. What is being decided

Whether per-part TensorRT engines beat our own kernels at the same precision,
per part, per regime -- with the CUDA-graph axis removed from **both** sides,
because TensorRT-RTX captures itself into a CUDA graph and a graph-vs-eager
number would credit TensorRT with the ~21x launch win #368 already measured.

Read `TARGET_SELECTION.md` first if you have not. The one thing to carry into
the run: the verdict cell is `trt_outer_graph / torch_graph`, and
`trt_*_graph / torch_eager` is in the JSON only so nobody quotes it.

## 1. Where everything is

```
repo        /spinning/wt-337-trt              branch feat/trt-microbench-337
scripts     scripts/trt_337/
artifacts   /spinning/gpu-battery-results/2026-08-05_trt_microbench_prep/
  engines/          32 AOT plans, CPU-built, real checkpoint weights:
                    INT8 per-stage + fold_bf16 / fold_fp16 whole-chain + one
                    fp32_ref diagnostic shape per rank
  pylibs/           tensorrt_rtx 1.6.1.120 + onnx, ISOLATED from every venv
  capability_probe.json
  shape_table.json
  mock_smoke_bench.json
  onnx/             structural artifact only, see export_onnx.py
```

**The library is deliberately not in any venv.** `tensorrt_rtx` lives under
`pylibs/` and is reached through `PYTHONPATH`. Do not `pip install` it into
`/spinning/htsglang-gpu/.venv` -- a serving window uses that venv.

## 2. Environment (both arms)

```bash
export ART=/spinning/gpu-battery-results/2026-08-05_trt_microbench_prep
export REPO=/spinning/wt-337-trt
export PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH="$ART/pylibs:$REPO/python"
# libnvrtc.so.13 for the triton quant kernel, same as every other run here
export LD_LIBRARY_PATH="/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
```

## 3. Pre-flight -- no lock, no card, run this first

```bash
$REPO/scripts/trt_337/mock_smoke.sh
```

Exercises every script end to end on the CPU with stub engines, INT8 and fold
variants both. Green on the desk on 2026-08-05: 10 points, 52 arm measurements,
tolerance and quality paths exercised on every point, with the fp16 chain
overflow surfaced as a NOTE. Re-run after any rebase -- a harness that has never executed is not
a harness.

## 4. Claim the cards

Per the rig's arbitration protocol (`/spinning/gpu-arb/`): take the per-card
lock, start the heartbeat, and release both when done. **Default assumption:
exclusive.** The measurement is a latency microbench at bs 1-8; another job on
the same card moves the SM clock and the result is then a number about the
other job. If the window owner explicitly agrees to share, record their
agreement in the result JSON's notes and treat every ratio as an upper bound.

Resolve the physical indices at runtime -- NVML order shifts between boots:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
```

The 5090 is the sm120 arm. Either 3080 is the sm86 arm.

## 5. Arm A -- sm120, the 5090, rank-0 shapes (~5 min)

```bash
CUDA_VISIBLE_DEVICES=<5090_idx> $PY $REPO/scripts/trt_337/microbench_trt.py \
  --engine-dir $ART/engines --rank 0 \
  --m 1,2,4,8 \
  --runtime-cache-dir $ART/rtcache \
  --out $ART/bench.sm120_5090.json
```

## 6. Arm B -- sm86, a 3080, rank-1 shapes (~5 min)

```bash
CUDA_VISIBLE_DEVICES=<3080_idx> $PY $REPO/scripts/trt_337/microbench_trt.py \
  --engine-dir $ART/engines --rank 1 \
  --m 1,2,4,8 \
  --runtime-cache-dir $ART/rtcache \
  --out $ART/bench.sm86_3080.json
```

## 6b. The fold arms -- included in arms A and B by default

Arms A and B above already run them; this section is what they mean and what to
look at. Three extra arms ride in the same rotation:

| arm | engine | what runs |
|---|---|---|
| `trt_fold_bf16_graph` | whole chain, one engine, bf16 weights | no activation quant at all |
| `trt_fold_fp16_graph` | same, fp16 weights | same |
| `trt_fp32_ref_graph` | `gemm_mlp_gate_up` only | **diagnostic, not a candidate** |

The fold engines were built at the desk alongside the INT8 ones and are already
in `engines/` (32 plans total).

**The crossover is the question.** The fold removes every activation-quant
kernel and pays 2x the weight bytes in a memory-bound GEMV. Read
`verdict.fold_bf16_over_torch_graph` across M = 1, 2, 4, 8 and find where it
crosses 1.0. Below the crossover the part is a fold candidate; above it, leave
it INT8. `verdict.fold_bf16_over_trt_int8` separates "the fold helps" from "the
engine helps" -- they are different wins and a per-part decision needs both.

**Quality is gated, not assumed.** Every point emits `quality`, with each path's
max-abs error against an exact fp32 reference built from the dequantized
checkpoint weights. The gate is
`quality.arms.*.at_least_as_accurate_as_deployed` -- a fold passes if its error
is no larger than the deployed INT8 path's. Check it; do not check the fold
against the deployed *output*, which would measure the deployed path's own
quantization error.

**Watch fp16 on the chain targets.** The desk mock produced a non-finite fp16
output on `chain_decode_layer` (4 GEMMs deep): fp16 has 11 mantissa bits but
only 5 exponent bits, max 65504, while bf16 carries fp32's range. The mock's
random weights exaggerate intermediate magnitudes, so the real-weight run is the
one that decides. If `quality.arms.trt_fold_fp16_graph.finite` is false, that
arm is a **result about fp16 in that chain**, not a broken run -- record it and
continue; bf16 is unaffected.

**Memory.** `engines/manifest.json` carries `weight_bytes_fold`,
`weight_bytes_int8` and `memory_multiplier_vs_int8` per engine (measured:
1.999x). Full-model fold does not fit the 3080s and is not the proposal -- the
proposal is a per-part stair. Report the winning parts with their MB so the
planner can price residency.

The fp32 arm is one shape, ~1 min, and exists only to separate TensorRT runtime
quality from the quantization constraints. At 4 bytes per weight it is
disqualified as a deployment candidate by memory; label it that way in any
summary.

## 7. Arm C -- cross-architecture control, optional (~5 min)

The AOT plans are portable: **the same `.plan` file runs on both cards.** So
rank-0 shapes can be run on a 3080, which separates "TensorRT is better at this
shape" from "TensorRT is better on this architecture" -- a control that the
per-arch build model could not have offered at all.

```bash
CUDA_VISIBLE_DEVICES=<3080_idx> $PY $REPO/scripts/trt_337/microbench_trt.py \
  --engine-dir $ART/engines --rank 0 \
  --m 1,4 --targets gemm_mlp_gate_up,chain_mlp_block \
  --runtime-cache-dir $ART/rtcache \
  --out $ART/bench.sm86_rank0shapes.json
```

## 8. Arm D -- classic TensorRT control, optional (~6 min incl. build)

NVIDIA states TensorRT-RTX has no performance downside against classic
TensorRT. That is a claim. Classic TensorRT (11.2.1.2) is already in the serving
venv, so the control costs one build plus one arm. Its builder needs a GPU, so
unlike everything else here the build is **inside** the window:

```bash
CUDA_VISIBLE_DEVICES=<5090_idx> $PY $REPO/scripts/trt_337/build_engines.py \
  --library classic --ranks 0 --out-dir $ART/engines-classic
CUDA_VISIBLE_DEVICES=<5090_idx> $PY $REPO/scripts/trt_337/microbench_trt.py \
  --engine-dir $ART/engines-classic --rank 0 --m 1,4 \
  --targets gemm_mlp_gate_up --out $ART/bench.classic_sm120.json
```

Classic TensorRT has no `create_runtime_config`, so `trt_native_graph` is not
runnable there; the harness records that as a note and carries on. If this arm
is skipped, say so in the result: **the claim is then unverified, not
confirmed.**

## 9. JIT warm-up and the runtime cache

The plans were built without a GPU, so the first execution on each card pays
TensorRT's JIT specialisation. The harness measures it (`jit.first_execution_seconds`
vs `jit.second_execution_seconds`), keeps it out of every timed region, and
serialises the resulting runtime cache to `--runtime-cache-dir`, keyed by
architecture and stage. A second window run finds `runtime_cache_preloaded:
true` and pays it once.

Report the first-run cost separately. "TensorRT is fast once warm" and
"TensorRT is fast" are different claims and a deployment cares about both.

Kernel specialisation strategy is `EAGER`, recorded in
`settings.kernel_specialization`, so no measurement lands on the generic
fallback kernel.

## 10. What to check before believing any number

1. `duration_rule_met: true` on every point. Each point runs >= 10 s measured.
2. `verdict.a2_floor_frac` -- the A-vs-A spread. If
   `verdict.inside_noise_floor` is true, that point decided nothing.
3. `quality.arms.*.at_least_as_accurate_as_deployed` on every fold arm, and
   `quality.arms.*.finite`. A fold that is less accurate than the deployed path
   is not a fold candidate, whatever its speed.
4. `tolerance.pass` and `tolerance.max_abs_diff`. Byte identity is not expected;
   the bound is 2e-2 relative. **A failure does not stop the run** -- an engine
   that is fast and wrong is a result, and it is recorded as one.
5. `clocks.before` / `clocks.after` -- SM clock, power, temperature, throttle
   reasons. Power targets on this rig are reduced (3080 200 W, 5090 400 W); a
   point that throttled mid-rotation is a point about thermals.
6. `environment.kernels_missing` empty, `environment.capability` matches the arm
   you think you ran.
7. **Did an INT8 tactic actually run?** A TensorRT loss caused by a silently
   chosen fp32 tactic is a different finding than a loss on equal footing. The
   engine inspector output is available via `TrtEngine.layer_information()`; if
   the verdict comes out badly against TensorRT, dump it before concluding
   anything.

## 11. Abort criteria

Stop and report rather than pushing on, if:

- **Any card is not exclusively yours** and the window owner has not agreed.
  Nothing measured under contention is worth the card time.
- **`tolerance.pass` is false on the simple GEMM targets (T1-T3).** The chain
  targets can drift a little through repeated bf16 rounding; a standalone GEMM
  cannot. False there means the engine is computing something else, and every
  timing after it is meaningless. Dump `layer_information()` and stop.
- **`jit.first_execution_seconds` exceeds ~60 s for one stage.** The documented
  AOT+JIT total is under 30 s for a whole model; a single GEMM taking longer
  means the JIT is falling back to something pathological.
- **A card throttles** (`clocks.*.throttle_reasons` non-zero) across a rotation.
  Let it cool, re-run the point; do not average across a throttle event.
- **`inside_noise_floor` is true on every point of an arm.** The arm is not
  resolving anything at these durations. Raise `--min-point-seconds` and
  `--target-ms` rather than reporting a null that is really an instrument
  limit.

## 12. After the run

- Both `bench.*.json` into `$ART/`.
- Record in the result notes: which cards, exclusive or shared, the NVML
  index -> card-name mapping you resolved, and whether arms C and D ran.
- The verdict sentence should name the cell: "trt_outer_graph / torch_graph =
  X at (target, M) on (arch)", never a ratio against eager.
- For the fold: report the crossover M per part, with the part's fold MB next to
  it. "gate_up folds below M=N at +83 MB per layer" is the sentence the planner
  can act on; "the fold is faster" is not.
- If TensorRT wins: the next question is integration cost, not more
  microbenchmarks. If it loses: `layer_information()` decides whether it lost on
  equal footing or on a bad tactic, and those lead to different next tasks.
