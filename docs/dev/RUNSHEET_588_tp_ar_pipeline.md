# RUNSHEET 588 -- token-slice pipelining of the TP all-reduce

Branch `feat/tp-ar-pipeline-588` (off `integration/r3-probe-next2` @ `c91cbbbb52`).
Desk slice is complete and hermetically tested; this sheet is the GPU window
that decides whether the lever pays.

Flag: `SGLANG_TP_AR_PIPELINE=1` (off by default).
Companions: `SGLANG_TP_AR_PIPELINE_MAX_SLICES` (8), `SGLANG_TP_AR_PIPELINE_MIN_TOKENS`
(256), `SGLANG_TP_AR_PIPELINE_SLICES` (0 = derive K from the measured cost model).

## What the mechanism does

`RowParallelLinear.forward` splits the token axis into K slices, computes
slice `i`'s GEMM on the compute stream, and issues slice `i`'s all-reduce on a
side stream while slice `i+1`'s GEMM runs. The join happens before the first
consumer, inside the same call. Arithmetic is unchanged: every output element
is still the sum of the same per-rank partials.

## The ceiling -- read this before judging the numbers

The consumer of a row-parallel output is the next statement (a layernorm or
residual add). There is no independent downstream compute, so the only
overlap partner is the layer's OWN GEMM. With per-slice compute `g = G/K` and
per-slice transfer `a = L + P/(K*B)`:

```
makespan(K) = (K-1) * max(g, a) + g + a
            = P/B + G/K + K*L          (transfer-bound, which is this rig)
baseline    = P/B + G + L
saving      = G - G/K - (K-1)*L        <= G
best        = P/B + 2*sqrt(G*L)        at K* = sqrt(G/L)
```

Three consequences, all of which the acceptance must respect:

1. **The transfer term `P/B` never moves.** The lever hides compute under the
   wire; it does not shrink the wire. The 1.2 s collective floor cannot fall
   below the pure transfer time plus one slice of compute plus the K launch
   latencies.
2. **The saving is bounded by the pipelined layers' own GEMM time**, not by
   the total compute of the forward. Prefill compute measured in window 5 is
   roughly 0.4-0.6 s against the 1.2 s wait (wait = 2-3x compute); the
   `o_proj` + `down_proj` GEMMs are a SUBSET of that. So the predicted band
   for the total prefill wait reduction is
   **0 ms .. (row-linear GEMM total) x (1 - 1/K) minus (K-1) x L per call**,
   with K ~ 6 and L ~ 30 us on this rig -- i.e. at best a low-hundreds-of-ms
   dent in a 1.2 s floor, not its removal.
3. **First-slice compute and last-slice transfer are never hidden.** That is
   the whole K trade: both exposures fall as 1/K while the launch overhead
   grows as K*L. The calibration minimizes exactly that sum.

If the measurement lands at this ceiling, the lever is exhausted and the
remaining floor is pure wire. That is a **PASS with a finding** -- it retires
the quality-neutral overlap question and hands the next move to transport
(barlink/smallbar) or to a coarser token split (see "follow-up" below).

## Arm 0 -- coverage check (do this FIRST, it can invalidate everything else)

The production model `qwen3_5.py` constructs `o_proj` and `out_proj` with
`reduce_results=False` (`python/sglang/srt/models/qwen3_5.py:352`, `:952`) --
those all-reduces are deferred to `LayerCommunicator`, and the MoE output
all-reduce lives in `layers/moe/fused_moe_triton/layer.py:2044`. The hook
built here sits in `RowParallelLinear.forward` and fires only for layers that
reduce in place. **Verify the hook fires before measuring anything.**

Procedure: boot with `SGLANG_TP_AR_PIPELINE=1`, send one long prefill, then
read the counters:

```python
from sglang.srt.distributed.tp_ar_pipeline import tp_ar_pipeline_stats
tp_ar_pipeline_stats()
# {'calls_pipelined': N, 'calls_unsliced': M, 'slices_issued': S,
#  'calibrated': True, 'calibration': {...}}
```

Equivalently, grep the serving log for the one-shot lines
`tp_ar_pipeline calibration:` and `tp_ar_pipeline: tokens=... -> K=`.

- `calls_pipelined == 0` -> the arm measured the baseline twice. Do NOT report
  a delta. Either switch the measurement vehicle to a model whose row linears
  reduce in place (dense Qwen3), or stop and take the follow-up ticket.
- `calls_pipelined > 0` -> record N and the derived K, then proceed.

## Arms

Same boot, same clocks, same model, back-to-back. Prefill point chosen by the
>= 10 s duration rule (a single prefill measurement must not be shorter than
the noise floor of the harness).

| Arm | Env | Purpose |
| --- | --- | --- |
| A/A | `SGLANG_TP_AR_PIPELINE` unset, twice | Own noise floor. Discard warmup, run back-to-back. Any A/B smaller than this spread is not a result. |
| OFF | `SGLANG_TP_AR_PIPELINE` unset | Baseline. |
| ON-auto | `SGLANG_TP_AR_PIPELINE=1` | K derived from the measured cost model. |
| ON-K2 | `SGLANG_TP_AR_PIPELINE=1 SGLANG_TP_AR_PIPELINE_SLICES=2` | Isolates "is any overlap happening" from "is K right". |
| ON-K8 | `SGLANG_TP_AR_PIPELINE=1 SGLANG_TP_AR_PIPELINE_SLICES=8` | Over-slicing: launch overhead should start eating the gain. If K8 beats ON-auto the calibration is wrong, and that is a finding about the fit, not about the mechanism. |

## Metric

The per-rank prefill line's **wait-by-family** decomposition (CollectiveClock).
Two families matter, and they are deliberately separated:

- `tp.all_reduce` -- baseline family. Present in OFF.
- `tp.all_reduce.pipe_wire` -- comm-stream busy time in the ON arms. This is
  **wire occupancy, not wait**; it will look large even when the overlap is
  working perfectly. Do not compare it against OFF's `tp.all_reduce`.
- `tp.ar_pipeline.join` -- **this is the exposed wait** of the ON arms,
  recorded on the compute stream. The headline comparison is
  `OFF tp.all_reduce` vs `ON (tp.ar_pipeline.join + any residual tp.all_reduce)`.

Also record: total prefill gpu-ms per rank, compute/wait split, tokens
prefilled, and the derived K from the calibration log line.

## Byte-identity gate (mandatory, before any perf claim)

Token-axis slicing does not change the arithmetic, but it does change the
GEMM's M and the collective's message size, and both cuBLAS kernel selection
and NCCL algorithm selection are size-dependent. Bitwise identity across the
slice boundary is therefore a KERNEL property that only the GPU can settle;
the hermetic suite proves the slicing logic, not this.

Gate: same prompt, greedy decode, fixed seed, OFF vs ON-auto vs ON-K2.

- Identical token ids -> byte-identity holds for this shape family; report
  the lever as quality-neutral.
- Divergent token ids -> record WHICH arm diverges. If ON-K2 already
  diverges, the cause is kernel/algorithm selection, not the pipeline; the
  lever then falls under the lossy-feature rule and is deferred, exactly like
  compressed collectives. Do not merge-enable in that case.

## Risk watchlist

- **Hang** (rank-divergent K). Everything feeding K is group-uniform and
  pinned by `test_derive_num_slices_signature_is_rank_uniform`. If a boot
  hangs at the first long prefill, capture `py-spy dump` on every rank before
  killing anything -- a rank stuck in a collective while another is in a GEMM
  is the signature.
- **Graph capture.** `plan_num_slices` returns 1 while a capture is active,
  so captured decode is untouched. If a captured PREFILL path is enabled in
  the future this needs re-checking.
- **Symmetric memory.** The hook declines when `is_allocation_symmetric()` is
  true; a boot with symm-mem on will simply show `calls_pipelined == 0`.
- **Quantized communications.** Declined for the same reason (the cost model
  does not describe a changing payload dtype).

## Follow-up (not this ticket)

If arm 0 shows `calls_pipelined == 0` on the production model, the mechanism
is sound but sits in the wrong place for that model. The extension is to let
`RowParallelLinear` run the sliced GEMM with `reduce_results=False` and hand
the join downstream as a handle on the tensor -- exactly the pattern barlink
already uses (`linear.py:2275-2296` issues, `communicator.py:567-597`
completes). That is a larger and riskier change (five all-reduce sites in the
communicator's decision tree, with double-reduce as the failure mode) and
must not be bolted on inside a window.

The coarser lever, if this one lands at its ceiling: slice the token axis at
the DECODER LAYER level rather than inside one linear, so a slice's
all-reduce overlaps the next slice's attention and gate/up GEMMs instead of
only its own projection. That is chunked prefill with overlap, and its
ceiling is the whole layer's compute rather than one GEMM's.
