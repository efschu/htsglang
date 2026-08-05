# RUNSHEET 597 -- deferred-join all-reduce at the LayerCommunicator seam

Branch `feat/tp-ar-pipeline-597` (off integration `d5da347708`).
Follow-up to #588, whose window-8 arm 0 came back `calls_pipelined == 0`.

Flags: `SGLANG_TP_AR_PIPELINE_DEFERRED=1` (off by default),
`SGLANG_TP_AR_PIPELINE_DEFERRED_MIN_TOKENS` (256). Independent of #588's
`SGLANG_TP_AR_PIPELINE`; the two hide a collective under different things and
may be run separately or together.

## What changed since #588

Window 8 measured the target family in production: **`tp.all_reduce`
932.2 ms over 129 calls** of a 96k-token prefill. It is not issued by a
row-parallel linear. `qwen3_5` builds `o_proj`/`out_proj` with
`reduce_results=False` and reduces the MoE output inside the MoE layer, so
#588's in-call hook never saw it.

This lever issues that same reduction on the comm stream at the site that
already owned it (`fused_moe_triton/layer.py:2041` `forward_impl`) and joins
at the first consumer inside `LayerCommunicator` (`prepare_attn`,
`prepare_mlp`, `postprocess_layer`). The reduction still happens exactly
once; only its position in wall-clock time moves.

## The recomputed ceiling

Let W be the compute between issue and join, G the producer compute a sliced
issue could additionally interleave (0 here -- the MoE producer is not
sliced), and `T_ar = L + P/B`.

```
baseline  = G + W + T_ar
pipelined = max(G + W, G/K + K*L + P/B)
saving    = min( G*(1 - 1/K) + W - (K-1)*L ,  T_ar )
```

Two changes against #588, where the bound was `G` alone:

1. **The overlap partner grew** from "the layer's own GEMM" to `G + W`. W is
   strictly additional, so this ceiling is higher by construction.
2. **The saving is now also capped by `T_ar`.** Once the whole collective is
   hidden there is nothing further to win. #588 could never reach that cap;
   this lever can. If the measurement lands on the `T_ar` cap, the family is
   fully hidden and the lever is DONE, not merely improved.

K is unchanged at 1 here: `derive_num_slices` is reused verbatim, and with
`G = 0` splitting a bare transfer only adds K launch latencies. Token slicing
pays only where the producer is sliceable; that is a property of the lever,
not a gap in the implementation.

**W is not predictable from the source.** It depends on which consumer joins
first, which depends on the layer order and on flags that move the reduce
point (`fuse_mlp_allreduce` skips `postprocess_layer` entirely and pushes the
join into the NEXT layer's `prepare_attn`, a strictly larger window). So the
code measures it:

```python
from sglang.srt.distributed.tp_ar_pipeline import tp_ar_pipeline_stats
tp_ar_pipeline_stats()["deferred_window_mean_ms"]   # this is W
```

**Read W first, then predict.** Expected saving per call is
`min(W, T_ar)`; with 129 calls and 932.2 ms of `tp.all_reduce`,
`T_ar ~ 7.2 ms/call`. So:

- `W >= 7.2 ms` -> the family can be hidden essentially in full; expect the
  wait-by-family line to collapse toward the join family.
- `W ~ 1-2 ms` -> expect ~15-30 % of 932 ms, i.e. 140-280 ms.
- `W < 0.5 ms` -> the lever is structurally exhausted at this join point. The
  finding is then that the reduce point and its first consumer are too close,
  and the next move is to widen the window (see follow-up), not to tune K.

## Arm 0 -- coverage check (FIRST, again)

Window 8 lost an arm to a hook that never fired. That specific outcome is now
pinned hermetically -- `test_coverage_moe_issue_fires_on_a_qwen3_5_shaped_config`
drives the real `FusedMoE.forward_impl` at production shape and fails if the
issue does not happen -- but the boot still has to confirm it end to end.

```python
tp_ar_pipeline_stats()
# {'deferred_issued': N, 'deferred_joined': N, 'deferred_declined': D,
#  'deferred_reduce_site_hits': 0, 'deferred_window_mean_ms': W, ...}
```

Acceptance before any timing arm:

- `deferred_issued > 0` -- the hook fired. If 0, stop; check
  `deferred_declined` (capture active, sub-gate token counts, or a
  non-reducing MoE config).
- `deferred_issued == deferred_joined` -- every issue was completed. A gap
  means a consumer path exists that does not go through the communicator, and
  that path is reading unreduced data. **Stop and report; do not measure.**
- `deferred_reduce_site_hits == 0` -- **hard gate**. Any non-zero value means
  a tensor with a pending handle reached an all-reduce site and the data was
  reduced twice. Abort the window and treat it as a correctness bug.

## Arms

Same boot, same clocks, same model, back-to-back. Prefill sized by the >= 10 s
duration rule against window 8's carry-forward of ~1104 prefill tok/s:
**11k-16.5k tokens** per prefill measurement.

| Arm | Env | Purpose |
| --- | --- | --- |
| A/A | deferred unset, twice | Own noise floor. Warmup discarded, back-to-back. Any A/B below this spread is not a result. |
| OFF | deferred unset | Baseline. `tp.all_reduce` should reproduce the window-8 shape (~7.2 ms/call). |
| ON-auto | `SGLANG_TP_AR_PIPELINE_DEFERRED=1` | The lever. K is 1 by derivation. |
| ON-K2 | `...DEFERRED=1 SGLANG_TP_AR_PIPELINE=1 SGLANG_TP_AR_PIPELINE_SLICES=2` | Both levers, in case a sliceable producer is also on the path. Separates "deferral works" from "slicing works". |
| ON-K8 | `...DEFERRED=1 SGLANG_TP_AR_PIPELINE=1 SGLANG_TP_AR_PIPELINE_SLICES=8` | Over-slicing control; launch overhead should start eating the gain. |

## Metric

Wait-by-family from CollectiveClock. Three families, deliberately separate:

- `tp.all_reduce` -- baseline family, present in OFF.
- `tp.all_reduce.pipe_wire` -- comm-stream busy time in the ON arms. **Wire
  occupancy, not wait.** Do not compare against OFF's `tp.all_reduce`.
- `tp.ar_pipeline.deferred_join` -- **the exposed wait**, recorded on the
  compute stream where a consumer blocks. Headline comparison is
  `OFF tp.all_reduce` vs `ON tp.ar_pipeline.deferred_join`.

Also record: `deferred_window_mean_ms` (W, the ceiling input), prefill gpu-ms
per rank, compute/wait split, tokens prefilled, and the issue/join counters.

## Byte-identity gate (mandatory, before any perf claim)

Deferring a reduction does not change its arithmetic -- same operands, same
backend, same message size, one call. The risk here is not rounding but
ORDERING: a consumer that reads the tensor without joining would see partially
reduced data, which shows up as wrong output, not as a small delta.

Gate: same prompt, greedy decode, fixed seed, OFF vs ON-auto.

- Identical token ids **and** `deferred_issued == deferred_joined` **and**
  `deferred_reduce_site_hits == 0` -> quality-neutral, report the delta.
- Divergent token ids -> a consumer is reading before the join. This is a
  correctness bug in join placement, NOT a lossy-feature trade; do not defer
  it under the lossy rule, fix the placement. Capture which consumer by
  bisecting the three entry-point joins.

## Risk watchlist

- **Consumed before join.** The one real hazard. Joins are planted at all
  three `LayerCommunicator` entry points; a model that touches the MoE output
  outside the communicator would bypass them. `deferred_issued ==
  deferred_joined` is the detector.
- **Double reduce.** Structurally excluded (the issue is only taken where the
  reduction was already owned) and guarded at all six reducing sites plus the
  two fused-layernorm paths by `note_reduce_site`. Counter must stay 0.
- **Graph capture.** The issue declines while a capture is active; captured
  decode is untouched.
- **Handle on a reshaped tensor.** The handle rides on the tensor object; a
  view or reshape between issue and join would drop it and the join would
  silently not happen. The issue/join counter equality catches this.

## Follow-up (not this ticket)

If W turns out too small to pay, the move is to WIDEN the window rather than
tune K: issue the MoE reduce before the final `combine`/slice steps of
`forward_local` (more of the producer's own tail becomes W), or run with
`fuse_mlp_allreduce` on so the join lands in the next layer's `prepare_attn`.
Both change where the tensor is first read and therefore need the same
issue/join accounting re-verified.

The remaining unhooked case is the reduce_results=False producer whose reduce
belongs to the communicator. Hooking it requires SUPPRESSING the
communicator's own reduce, which is the double-reduce failure mode --
deliberately out of scope here, and `note_reduce_site` is already planted so
that attempt fails loudly in the test suite instead of quietly returning
doubled activations.
