# #613 -- prefill CUDA graph under barlink: DETERMINATION

Date: 2026-08-17. Hermetic (`CUDA_VISIBLE_DEVICES=""`), no boots.

**Verdict: there is no barlink deactivation to reverse. The barlink defect
#613 named was fixed and merged eleven days ago, the barlink measurement was
already TAKEN, and it came back as a REGIME SPLIT rather than a yes/no. The
actuator that split implies -- a per-batch gate -- had never been built. It is
built here.**

## (a) Where is the deactivation?

**Not in barlink, and not in any barlink-conditioned code.** Grepped every
place that can force `cuda_graph_config.prefill.backend = Backend.DISABLED`:

| rule list | file:line | barlink term? |
|---|---|---|
| `_disable_piecewise_cudagraph...` (tc_piecewise) | `server_args.py:9442-9511` | none |
| `_disable_breakable_cudagraph_if_incompatible` (BCG) | `server_args.py:9513-9556` | none |
| `_disable_full_prefill_cudagraph_if_incompatible` | `server_args.py:9558-9569` | empty rule list, inert |

`git log -S'barlink' -- server_args.py cuda_graph_config.py` returns exactly
one commit -- `501ccb14b3`, the #613 fix itself -- confirming barlink was never
wired into either rule list, past or present.

What actually makes production prefill eager is an **upstream,
transport-agnostic** rule: `("multimodal model", lambda:
self.get_model_config().is_multimodal)` at **`server_args.py:9546`**, from
upstream PR #29458. It fires because the served checkpoint really is
multimodal -- verified live rather than assumed: the running server reports
`has_image_understanding: true`, and
`Qwen3.8-27B-INT8-yarn1.5/config.json` is a `Qwen3_5ForConditionalGeneration`
carrying `vision_config` with `deepstack_visual_indexes` and **333
`model.visual.*` tensors**. The rule is correct and merely coarse: a
whole-server switch for a per-batch property, while production traffic is
text-only.

(NOTE_515 cited this rule at `:8511`; at HEAD it is `:9546`. Another instance
of the symbol-anchor rule.)

The prefill graph is otherwise ON by default: `default_prefill_backend()`
returns breakable on CUDA (`cuda_graph_config.py:95`).

## (b) Is the stated reason still true?

**No. Both named defects are fixed and both fixes are ancestors of HEAD.**

| defect, as named | fix | in lineage? |
|---|---|---|
| "the 1blk restriction for above-threshold collectives active during graph capture" -- `graph_grid_default()` read the release switch through `_is_on()` (unset = OFF) while the canonical `graph_enable_set()` has defaulted ON since #369, "costing 16.1% prefill throughput once prefill is captured" | `501ccb14b3` (#613) | yes |
| "a tripped spin kernel must not kill the CUDA context" | `b001d102fa` (#583) | yes |

Adjacent and stronger than either: **#632** (`b42405c0ee`) found the barlink
bar1 mesh/a2a peer barrier deadlocking **inside graph replay** -- the one defect
class that would specifically break a captured arm -- and replaced it with a
device-side, capture/replay-safe consumption-ack barrier. Merged as
`c6f5b57c3f` on a **2h26m+ amplified-load soak: zero aborts, zero FREEZE, 9.2M
ack-barrier rounds with strict cross-rank watermark growth through graph
replay**.

So barlink is capturable today, by default, and its capture is proven:
`capturable_transports()` = `{device, host}` always, plus `{bar1, matrix}`
while the #369 release switch is on (its default), and the cooperative-launch
question was answered on real cards on 2026-08-01 (10/10, including the `grid`
case). The only barlink transports refused under graphs are the host-staged
ones, which is correct -- a host-synchronising collective inside a capture
raises `cudaErrorStreamCaptureUnsupported`.

## (c) What prefill needs that decode does not

| seam | state at HEAD |
|---|---|
| raw-graph capture must enter `model_capture_mode()` (`disable_dispose_tensor`, `get_is_capture_mode()`) -- three capture-time properties were silently False on the prefill path | **fixed** (#356), `prefill_cuda_graph_runner.py:805-841`, with the rationale in-tree |
| breakable (BCG) is the prefill route; decode is full | as designed, `cuda_graph_config.py:95` |
| rank-divergent-collective discipline during capture | live, `parallel_state.py:3438`, `:3494` (#431/#616B/#645 family) |
| host-staged transport refused under capture | live, `_enforce_cpu_transport_needs_eager`, `parallel_state.py:413` |

None of these is open.

## The finding that changes the task: the measurement was already taken

`3b4526c4ac` (2026-08-05, on the unmerged report-only branch
`feat/prefill-graphs-reenable`) ran prefill-eager vs prefill-graphs **under
barlink** -- production's transport -- interleaved, ms per fixed unit of work,
each point against its own A-vs-A floor:

| point | eager | graphs | paired delta | floor | verdict |
|---|---|---|---|---|---|
| 1900 single-stream | 1196.3 ms | 1320.6 ms | **+10.25% slower** | 0.12% | REPORTABLE |
| 256 x4 concurrent | 230.1 ms | 219.0 ms | **-4.62% faster** | 1.99% | REPORTABLE |

All three pairs agreed in sign at both points; the deltas cleared their own
floors by 85x and 2.3x. The user's hypothesis holds, **and only in the regime it
predicted**: the captured prefill pays where the work is launch-train bound and
loses where it is GEMM bound.

Its own recorded conclusion: *"the answer is not on/off but workload-dependent,
and a blanket flag is the wrong shape. `can_run_graph` is already a per-batch
decision (`prefill_cuda_graph_runner.py:615`), which is where a regime gate
would live."*

**That gate was never built.** The branch carries report commits and no code.

## What is built here

`model_executor/runner/prefill_graph_regime.py` -- `regime_permits_graph`,
consulted from `can_run_graph` **last**, after every correctness check, so a
performance gate can only ever refuse a batch the correctness checks admitted,
never admit one they refused.

**It permits only the regime that was measured to win** (>= 2 in flight, at or
below 256 tokens/request) and refuses everything else BY NAME. The unmeasured
middle is refused **as unmeasured**, not as slow:

> `800 tokens/request is above the measured win point (256, -4.62%) and below
> the measured loss point (1974, +10.25%). The sign in between is UNMEASURED,
> so the graph is refused here rather than admitted on an interpolated
> crossover. Narrowing this needs a measurement, not an edit.`

Two points are not a curve. A gate that interpolated a crossover would be
inventing the number that decides every borderline batch -- the fitted-cell
mistake. Narrowing the refusal is a measurement, not an edit.

**DEFAULT OFF** (`SGLANG_PREFILL_GRAPH_REGIME`): with the lever unset the
verdict is always permit and `can_run_graph` is byte-identical to today.

12 hermetic tests, red first, mutation-proven: reinstating the blanket-flag
shape (dropping the single-stream refusal) reds exactly the three tests that
assert it. One test pins the WIRING -- that `can_run_graph` actually calls the
gate -- because a gate nothing calls is the #349 defect in another costume.

## What is NOT unblocked

**The content gate is still open, and it is the real blocker for production.**
Window 4 measured content divergence at **1/8 under barlink against 4/8 under
NCCL**, but the gate ran on pair 1 only, so there is **no barlink
eager-vs-eager content floor** -- the 1/8 is not yet distinguishable from
barlink boot noise and is recorded that way, not as an improvement. Determinism
arms did not close it either (1/8 with, 1/8 without). NOTE_515 §7's
recommendation stands until that floor exists.

**And the multimodal rule still fires**, so production has no prefill graph to
gate at all. Loosening a whole-server rule for a per-batch property is an
upstream design decision plus GPU evidence; this determination does not take
it.

So the honest state is: barlink is not the blocker and has not been since
2026-08-06; the regime actuator now exists and is off; the content floor and
the multimodal rule are what remain.

## Boot validation, turnkey

`bench/613/run_613_regime_gate.py`. Window 4 settled *whether* prefill graphs
pay; it could not test an actuator, because none existed. This validates the
actuator, and the question is narrow and falsifiable:

> with the gate ON, does the long single-stream point stop paying the
> captured-path penalty, while the short concurrent point keeps its win?

That needs **three** arms, not two -- EAGER (reference), GRAPH (graph on, gate
off = window 4's treatment), GATED (both on) -- each a separate boot, because
the prefill backend and the lever are both read at load time. The gate passes
iff, against EAGER, GATED sits at eager on the long point (it refused) and
keeps GRAPH's win on the short one (it permitted).

EAGER is **re-measured, never quoted from window 4**: those absolutes were
taken under specific power caps and a different checkpoint generation, so
citing them as this run's reference would be the borrowed-baseline mistake.
Every arm is measured in the same session against this session's own floor.

`--self-test` is hermetic: 18 checks, 5 of which assert the judge REJECTS --
a gate that did not refuse the long point, a gate too strict to keep the win, a
short point that got slower, and a win that does not clear its own floor.
Mutation-proven: forcing `long_ok = True` reds exactly the two checks that
catch a non-refusing gate. It also cross-checks the SHIPPING gate against its
own workload points, so the window cannot validate a gate other than the one in
the tree. Pinned by
`test/registered/unit/model_executor/test_regime_gate_runner_613.py`.

## Window items filed

1. **Barlink eager-vs-eager CONTENT floor** -- the missing control. Without it
   the 1/8 number cannot be read. Dep: none, it is a re-run of window 4's
   content arm with both pairs.
2. **Regime-gate A/B on the production traffic mix** -- whether the mix nets
   out positive, which window 4 explicitly left unmeasured. Dep: item 1, and a
   boot with the prefill backend forced on (`--cuda-graph-backend-prefill
   breakable` locks the phase and bypasses the compatibility rule, so no source
   change is needed).
3. **The crossover between 256 and 1974 tokens/request** -- currently refused
   as unmeasured. Measuring it is what narrows the gate.

`tests/prefill_graphs/window3_boot_floor.sh` had a stale precondition reason
pointing at a "barlink-583 repro window" that is nowhere recorded; it now names
the #622/#632 soak and window 4's 8 clean barlink arms, records that
precondition 1 (`b001d102fa`) is satisfied, and carries a scope note that the
stage is now a confirmation run rather than the primary experiment. The
operator attestation itself is KEPT -- a script may not grant itself one -- and
a third precondition was added: the BG arm captures, so a host-staged transport
is refused before any card is touched rather than faulting mid-boot as an
unrelated CUDA error.
