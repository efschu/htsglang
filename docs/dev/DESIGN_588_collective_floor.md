# DESIGN #588 — the ~1.18 s prefill collective floor: can it be hooked?

Desk decision, base `4a16043d1a` (`train/0817-control`). No boot, nothing
executed against a server. **No live wiring is proposed here**; the recommended
option is review+boot gated and named as such at the end.

The floor under discussion: `tp.all_reduce` ~900 ms + `dcp.all_reduce` ~260 ms
in a production prefill, both FLOORS rather than averages.

## 1. The mechanism triple, at code

| piece | file:line | what it is |
| --- | --- | --- |
| #597 issue site | `layers/moe/fused_moe_triton/layer.py:2062` | the ONLY `issue_deferred_all_reduce` caller in the tree |
| joins | `layers/communicator.py:675`, `:873`, `:891` | `join_deferred(hidden_states)` — producer-agnostic: joins whatever arrives carrying a handle |
| double-reduce guard | `distributed/tp_ar_pipeline.py:888` `note_reduce_site`, planted at `communicator.py:707`, `:716`, `:1203`, `:1239` | counts, warns, and JOINS so the data is deterministic rather than racing |
| #588 lever | `layers/linear.py:2288` | `pipelined_row_all_reduce` at `RowParallelLinear` |

## 2. Why the dense path has no issue site — the question, answered in two parts

The brief asked whether this is historical scoping or a real hazard. **It is
both, and which one depends on WHICH dense site you mean.** Conflating the two
is what makes the question look unanswerable.

### At the communicator (`communicator.py:1204`) — a REAL HAZARD

#597's safety argument is stated at its own issue site
(`fused_moe_triton/layer.py:2052-2055`) and it is precise:

> "The SAME single reduction happens either way — **nothing downstream reduces
> this tensor**, so nothing downstream has to be suppressed and a double reduce
> cannot arise from this change."

That property is exactly what `communicator.py:1204` does not have: the line
immediately after the guard is `attn_tp_group.all_reduce(hidden_states)` — the
COMMUNICATOR owns this reduction. Issuing from a producer upstream of it would
require suppressing it, and `note_reduce_site`'s docstring names that precise
case as the reason it exists:

> "it exists so that a future change which moves the issue to a producer whose
> reduce belongs to the communicator FAILS LOUDLY in the test suite instead of
> quietly returning doubled activations."

So the absence of an issue site here is a designed refusal, not an oversight.

### At `RowParallelLinear` (`linear.py:2338`) — SCOPING

The non-DP, non-quantized branch calls
`tensor_model_parallel_all_reduce(output_parallel)` under
`self.reduce_results`. That is a **producer-owned** reduce, structurally
identical to the MoE site #597 hooked. Nothing about it is hazardous in the way
the communicator site is; it simply was never hooked, because #597 was aimed at
the reduce that dominated the measured window on a **MoE** model (932.2 ms over
129 calls). On a DENSE model there is no MoE layer, so #597's one issue site
cannot fire at all — which is precisely why the lever reads as "inert" here.

## 3. #588's own lever is structurally inert — and not merely flag-gated

`linear.py:2293` requires `not is_allocation_symmetric()`, and
(`layers/dp_attention.py:309`):

```python
def is_allocation_symmetric() -> bool:
    return not is_dp_attention_enabled() or is_dp_max_padding()
```

On the served config — dense model, plain TP, **no DP attention** —
`is_dp_attention_enabled()` is False, so this returns **True**, so
`not is_allocation_symmetric()` is False. **#588 cannot engage on this
configuration even with `SGLANG_TP_AR_PIPELINE` set.** The inertness is an
eligibility property, not an unset flag, and the module says why: the pipeline
requires the symmetric-memory context to be DISABLED, and plain TP is exactly
where it is enabled.

That rules #588 out as the route to this floor. It is not a tuning question.

## 4. The three options, with their failure modes

### (a) Suppress the communicator-owned reduce, issue from a new dense site

**Rejected.** It requires the suppression to be exactly co-extensive with the
issue, across every path that reaches `communicator.py:1204`. The failure modes
are asymmetric, and that asymmetry is the argument:

* **issue without suppression** → double reduce. GUARDED: `note_reduce_site`
  counts it, warns, and joins, and the existing falsifier proves the catch
  works.
* **suppression without issue** → activations that were never reduced.
  **NOT guarded by anything in this mechanism.** The tensor carries no handle,
  so `note_reduce_site` returns it untouched at `:902`; there is no counter for
  "a reduce that should have happened and did not". That is the silent-wrongness
  direction, and it is the one with no net.

Taking (a) means accepting an unguarded failure mode in order to reach a floor
that (b) can reach without one.

### (b) Extend #597's issue mechanism to `RowParallelLinear` — RECOMMENDED

Same pattern, new site, **no suppression anywhere**: the call that would have
reduced is the call that defers, which is the whole of #597's safety argument.
Two things already exist and do not need building:

* the JOIN half is producer-agnostic — `join_deferred` at
  `communicator.py:675/873/891` joins whatever arrives with a handle;
* the guard and its falsifier are already in place (§5).

**The trade, named:** (b) is safe only if #597's property — *nothing between
the issue and the join reduces this tensor again* — holds for the dense tensor,
and that is a claim about the dense forward path that this desk has NOT
established. It is not assumed here. It is the evidence a build must produce,
and `note_reduce_site` plus the existing falsifier are the instrument that
would catch it if it is false.

### (c) Refuse — document the floor as irreducible

The honest fallback **if and only if** (b)'s property cannot be established.
It is not the verdict today, because nothing found in this reading forbids (b);
what is missing is a proof, not a mechanism.

## 5. The falsifier already exists — do not rebuild it

The brief asked for a hermetic test that fires on the double-reduce mode. It is
already built: `test/registered/unit/distributed/test_tp_ar_deferred_join.py:258`
`test_double_reduce_falsifier_can_fail`, with
`test_a_reducing_site_never_sees_a_pending_handle` (`:238`) as the zero-side
invariant and `test_every_communicator_all_reduce_site_is_guarded` (`:453`)
pinning that every site carries the guard.

It is also **stronger than what this brief specified**: besides asserting the
counter fires and the handle is cleared, it demonstrates that the second
reduction *would* corrupt the result (`doubled != reference`) — so the harm is
visible, not merely counted. 51 tests across the two #588/#597 suites pass on
this base.

Extended rather than rebuilt, per the prior-art rule.

## 6. What is review + boot gated

Everything in (b). Specifically, a build would owe:

1. the "nothing downstream reduces this tensor" property ESTABLISHED for the
   dense path between `linear.py:2338` and the communicator joins — by trace,
   not by analogy with the MoE site;
2. a coverage test proving the new issue site actually FIRES on a dense
   config (the existing suite has exactly this shape for MoE at `:378`, with
   its own can-fail at `:422` — copy that pattern, since a lever that never
   fires measures the baseline twice);
3. a boot, with `tp_ar_pipeline_stats()["deferred_issued"]` non-zero and
   `deferred_reduce_site_hits` still **zero**, on the served config;
4. the measured delta against the ~900 ms `tp.all_reduce` floor — the point of
   the exercise is the floor, and an overlap lever that engages without moving
   it is a finding too.

Decision on whether to build stays with the operator and review.
