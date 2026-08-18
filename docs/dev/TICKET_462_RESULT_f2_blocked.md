# TICKET 462 RESULT — eager control measured; the breakable route does not build

Window 2026-08-04, `/spinning/gpu-battery-results/2026-08-04_dsv4f_window/`.

**Power state:** 3080s 200 W (default 320), 5090 400 W (default 575). Not
comparable with any full-power anchor from an earlier day.

TICKET_462 / #494 state that neither the breakable route nor the break-cost
instrument had ever run on a card. This window is the first attempt.

---

## 1. The eager control — measured, and it is a good control

| | 462_eager |
|---|---|
| ready | 331 s |
| **decode ms/round** (bs=1, ctx 940) | **131.475** |
| A-vs-A floor, ms/round | **0.401 %** |
| determined | 5/8, answers identical to every other arm |
| chatprobe | template applied, negative control differs |

Cross-boot reproducibility is the useful result here. The `470_a_base` arm ran
the same configuration in the same window and measured **131.353 ms/round**.
Two independent boots, **0.09 % apart**. Combined with A-vs-A floors of 0.33 %,
0.40 % and 0.72 % on three separate arms, this establishes the real measurement
band for DSV4F decode on this rig at this power state: **well under half a
percent.**

That has a consequence for #470 — see §3.

---

## 2. The breakable route — REFUSED, and not by its own gate

The route was reached: `validate_breakable_boot` passed, the boot got as far as
graph capture, and the two capture-error strings the arm checks for
(`cudaErrorStreamCaptureUnsupported`, `cudaErrorStreamCaptureInvalidated`) were
absent. It died inside the capture itself:

```
File ".../runner_backend/breakable_cuda_graph_backend.py", line 143, in capture_one
File ".../runner_backend/breakable_cuda_graph_backend.py", line 189, in _alloc_full_buffer
    raise TypeError(f"Unsupported BCG output type: {type(output)}")
TypeError: Unsupported BCG output type: <class 'sglang.srt.layers.logits_processor.LogitsProcessorOutput'>
```

`_alloc_full_buffer` (and its siblings `_output_rows` and `_slice_output`)
handle exactly four shapes: a bare tensor, `PPProxyTensors`, `tuple` and `list`
(`breakable_cuda_graph_backend.py:165-205`). The model forward on this path
returns a `LogitsProcessorOutput`, which is none of them, so the backend cannot
allocate its replay buffer.

### Why this was NOT fixed in-window

The obvious fix is to teach the three methods a `LogitsProcessorOutput` branch.
It was deliberately not attempted, and the reason is the failure mode rather
than the effort: `LogitsProcessorOutput` is a structured output whose fields do
not share a leading dimension — some are per-token, some are per-request, some
are optional and absent depending on capture mode. `_slice_output` slices
`[:num_tokens]`. A branch that maps those fields wrongly **would not raise**; it
would return a buffer of the right shape holding the wrong rows, and the arm
would produce plausible logits and plausible ms/round numbers that are silently
incorrect.

That is precisely the silent-wrongness class this repo's rules single out, it
would land in the decode path of a speculative feature, and it is not something
to write against a restore deadline with no reference output to check against.
The correct sequence is a desk pass over `LogitsProcessorOutput`'s field
semantics plus a byte-identity falsifier against the eager control, then a boot.

### WRITTEN AFTER THE WINDOW (#535/B2), with the desk pass first

The contract was derived from code, not from the dataclass comments:

* `DecodeCudaGraphRunner.execute` is the only consumer of a decode replay
  output and slices `next_token_logits`, `full_logits`, `hidden_states` and
  `cross_aux_hidden_states` with the SAME `[: raw_num_token]`
  (`decode_cuda_graph_runner.py:2112-2144`). That is the downstream statement
  that the four share a leading dimension here.
* They share it because `_get_pruned_states` sets `pruned_states =
  hidden_states` and `sample_indices = None` in `decode_or_idle` /
  `target_verify` / `draft_extend_v2` (`logits_processor.py:585-596`) — the
  sampled logits are never gathered down to one row per sequence.
* The one place the fields genuinely DISAGREE is prefill:
  `prefill_cuda_graph_runner.py:1194` slices `next_token_logits` by `raw_bs`
  while `hidden_states` goes by `raw_num_tokens`. That case cannot reach this
  branch — with a prefill BCG the runner captures the LAYER MODEL body (a bare
  tensor) and runs the LM head and logits processor eagerly outside the graph
  (`:1103-1130`).
* Parts 2 and 3 of the dataclass (the sampler's logprob fields, the
  prefill-only input-logprob fields) and `customized_info` are host-side and
  never rewritten by a replay. They are REFUSED by name, not passed through:
  a replay buffer would serve them frozen at capture-time content with the
  right type and no exception.

So the branch is an allowlist of five per-token tensor fields
(`_LPO_TOKEN_DIM_FIELDS`), a by-name refusal for everything else, and a refusal
when the present fields disagree on their leading dimension. Nothing is
guessed; the underdetermined case raises.

**A second, independent defect was found in the same layer and fixed with it.**
The shared output buffer was sized from `shape_key.size`, which for a decode
runner is the BATCH size (`_capture_graph_size`,
`decode_cuda_graph_runner.py:682`), while the body's output is indexed by
TOKENS — `bs * num_tokens_per_bs` under a non-ragged speculative verify. One
row per sequence for a per-token output truncates every draft position but the
first, with no error, on exactly the F2 arm. The budget now takes the body's
own leading dimension where it is larger; captures run largest-first, so the
first one sets the high-water mark, and a later shape that needed more refuses
instead of truncating.

Tests: `test/registered/unit/model_executor/test_bcg_logits_output_buffer_462.py`,
18 hermetic arms — per-field value round-trip (a swapped mapping between two
same-shaped fields is caught on VALUES, which shape checks cannot do), the
disagreement refusal with its spread precondition, the by-name refusals, the
row budget, neutrality for the four pre-existing output shapes, and a
mock-side capture→store→replay smoke that ends on a discrimination check
(the stored outputs must be VIEWS onto the shared buffer, or a replay would
serve capture-time logits). TWO executed can-fail mutations: removing the
leading-dim refusal → red; taking the row budget from the graph key → red.

**Still not a measurement.** No boot, no replay on a card, no crossing count,
no F2 verdict. `43 crossings/round` remains an unverified desk figure. The next
window's falsifier is the F2 arm rerun: the capture must now complete, and the
first thing to check after it does is that the breakable arm's greedy output is
byte-identical to the `462_eager` control (131.475 ms/round, floor 0.401 %) —
identical text is what rules out the wrong-rows failure this branch was held
back for, and it is a stronger check than any shape assertion.

### What is now known that was not known before

* The breakable route's **configuration** path is sound end to end:
  `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable` plus
  `--cuda-graph-backend-decode=breakable --cuda-graph-backend-prefill=disabled`
  passes `validate_breakable_boot` and reaches capture. No config work is
  outstanding.
* The blocker is a single missing output-type branch in the BCG buffer layer,
  not anything about offload, graph breaks, or the #494 instrument.
* **The #494 break-cost instrument remains unexercised on hardware.** It never
  ran, because the route never captured. No crossing count, no per-crossing
  cost, no F2 verdict. `43 crossings/round` is still an unverified desk figure.

### A script defect fixed along the way

The eager control arm passed `--cuda-graph-backend-decode=disabled
--cuda-graph-backend-prefill=disabled` and was refused by the offload path's own
guard, which demands `--disable-cuda-graph` by name:

```
RuntimeError: MoE expert-offload / routing-trace ... requires --disable-cuda-graph
```

Corrected to `--disable-cuda-graph`, which is also what the proven base recipe
uses — so the control arm is now the recipe unmodified, which is what a control
should be.

---

## 3. Consequence for #470's headline number

`TICKET_470_RESULT_first_boot.md` reported the residency cut as **+1.41 %
against a governing floor of 6.443 %**, i.e. not resolvable, because the rule is
to gate on the larger of the two arms' own floors and `a_base` measured 6.443 %.

This window now has four arms' floors: **0.33 %, 0.40 %, 0.72 % and 6.443 %**,
plus a 0.09 % cross-boot reproduction of the `a_base` configuration itself. The
6.443 % figure is an outlier, and `a_base` was the arm that also ran three
1000-token accept generations.

Revised reading, carried in both documents: **the residency cut most likely
costs ~1.3-1.4 % of decode ms/round — small, but real** rather than zero. The
conservative "inside the floor" statement is what the rule produces mechanically
from `a_base`'s own floor; it is retained as the formally-defensible bound, but
the better-supported estimate is a small real cost. Either way the conclusion
for the R1 gate is unchanged: the cost side is small enough that any positive
return from a working draft arm clears it.

---

## 4. #535 UNBLOCK VERDICT (2026-08-18): F2 is a pure window ticket

The §2 blocker branch is BUILT and contained in the comp4 lineage
(`921d63defc`, ancestry verified): `d64cf27147` adds the
`LogitsProcessorOutput` branch exactly as the desk pass above specified —
the five-field allowlist (`_LPO_TOKEN_DIM_FIELDS`), by-name refusal of the
host-side parts, refusal on leading-dimension disagreement — plus the
second row-budget defect (batch-sized buffer under token-indexed output).
`test_bcg_logits_output_buffer_462.py`: 18 passed here.

F2 re-run needs NO code. Window form: the §1 eager control recipe
(`--disable-cuda-graph`, the corrected control) against the breakable arm;
acceptances: capture completes past `_alloc_full_buffer` (the §2 traceback
absent), the #494 break-cost clock emits per-crossing numbers
(`breakable_cuda_graph` census), determined-answer identity vs the eager
control (the byte-identity falsifier the desk pass demanded), ms/round
within the window's own A-vs-A floor discipline. The `43 crossings/round`
desk figure gets its first hardware verdict.
