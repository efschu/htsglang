# #138 -- Adaptive draft length on MultiLayerEagleWorkerV2: DETERMINATION

**Verdict: already DELIVERED. Nothing to build. This ticket is closable.**

The determination was asked to find a partial wiring and a possibly-silent
refusal, then compose the smallest funded slice. It found neither gap: the
compose already shipped, and the code names the ticket while doing it.

## The finding, in one place

`adaptive_spec_params.py:225-231`, inside `adaptive_unsupported_reason`:

> `enable_multi_layer_eagle` used to be rejected here ("MultiLayerEagleWorkerV2
> does not implement adaptive"). It does now (#138): the worker implements the
> AdaptiveSpecWorker protocol, the ladder ceiling is bounded by the loaded MTP
> layer count, and `MultiLayerEagleWorkerV2._assert_adaptive_supported` enforces
> the multi-layer-specific constraints (candidate steps >= 1, ceiling <=
> `num_nextn_predict_layers`) with a hard error instead of a silent fallback.

Every load-bearing claim in that comment was verified at code rather than
taken on the comment's word.

## The four questions

**(a) Where the controller lives, and its worker-facing contract.**
The controller is `python/sglang/srt/speculative/adaptive_spec_params.py`
(`AdaptiveSpecWorker` decisions, `RungMetrics`, `AdaptiveSpeculativeParams`).
Provenance matches the standing memory and `FEATURE_CATALOG.md:1023`: adaptive
draft length is an **upstream base**, and the fork carries only the
Aufsaetze (graph-offload, high-accept ladder, frozen-MTP, hetero determinism).

The worker-facing contract is narrow and explicit -- a two-method Protocol at
`adaptive_runtime_state.py:187`:

| method | role |
|---|---|
| `build_adaptive_runtime_state(...) -> SpecRuntimeState` | worker exports its swappable state |
| `apply_runtime_state(state) -> None` | controller swaps a chosen rung back in |

**(b) Does MultiLayerEagleWorkerV2 consume it?**
Yes -- **fully, not partially**. The suspected partial wiring (the #421
detector-C shape) is not what the 41 `adaptive` references are. The worker
implements both protocol methods, at
`multi_layer_eagle_worker_v2.py:982` and `:1050`, plus its own admission check
`_assert_adaptive_supported` at `:949`.

**(c) What structurally differs for multi-step drafting.**
The real multi-layer-specific constraint is not "per-step length choice vs
one-shot k" -- it is that **the ladder ceiling is a weight-bound quantity**.
A multi-layer worker can only draft as deep as the MTP layers it actually
loaded, so the ceiling is bounded by `num_nextn_predict_layers`, and a bad
ceiling **must fail before the weight load** rather than at first decode.
That ordering is the substantive difference, and it is already honoured.

**(d) Is there a refusing gate today?**
There is no silent ignore -- so this is **not** a #505a-class finding.
The gate stopped naming the multi-layer worker (the mention that remains is a
historical comment, never a returned reason), and the worker refuses bad
configs with a hard `ValueError` that names the remedy
(`--speculative-adaptive-config`). The gate also did not open generally: it
still refuses `enable_two_batch_overlap`, `enable_pdmux`,
`enable_dp_attention`, and `speculative_eagle_topk != 1`.

## What this deliverable adds

Closure is the deliverable, but closure alone would leave the delivery
undefended -- and its regression mode is **silent**. If the refusal were
reinstated, or either protocol method dropped in a refactor, adaptive would
simply stop applying on this worker with no error at all: exactly the
#505a shape this ticket was sent to look for, arriving later by regression
instead of by omission.

So the delivery is pinned from three directions in
`test/registered/unit/speculative/test_adaptive_multilayer_138.py` (7 tests):
the gate must not refuse multi-layer in live code (while still refusing what
it should), the worker must keep satisfying the protocol, and its constraint
check must keep raising rather than falling back.

Can-fail proof (both mutations applied and were asserted to apply):
reinstating the refusal as live code fails
`test_multi_layer_is_not_an_unsupported_reason`; renaming a protocol method
away fails `test_it_implements_both_protocol_methods`. Each mutation failed
exactly its own test, 6 others still passing.

## Scope note

No GPU measurement is filed. Nothing here needed one: every question was
answerable at code, and the acceptance signal for the *delivery* was already
banked when #138 was built. Whether adaptive draft length is a throughput
*win* on the multi-layer worker is a separate, unasked question.
