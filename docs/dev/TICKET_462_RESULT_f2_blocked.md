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
