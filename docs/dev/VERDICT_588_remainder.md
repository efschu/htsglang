# #588 remainder -- is it delivered by siblings? DETERMINATION

Date: 2026-08-17. Hermetic (`CUDA_VISIBLE_DEVICES=""`), no boots.

**Verdict: NO. Both sibling levers are built, tested, and INERT on the serving
configuration. The dominant term of the socket is untouched, and the #597
runsheet attributed it to a producer the measured model does not have.**

## (a) What the 1.2 s socket actually decomposes into

Measured, not recalled: read back from the window-8 record
(`/spinning/gpu-battery-results/2026-08-05_window8/boot_A2_588_coverage.log`,
49 prefill batches per rank) with `bench/588/run_588_socket.py`.

| rank | compute (med) | wait (med) | wait/compute | range |
|---|---|---|---|---|
| TP0 | 292.6 ms | 1473.9 ms | **4.91x** | 0.19 .. 7.87 |
| TP1 | 490.0 ms | 1272.0 ms | 2.48x | 0.20 .. 5.09 |
| TP2 | 483.5 ms | 1283.5 ms | 2.57x | 0.19 .. 5.39 |

| family | class | per-rank median |
|---|---|---|
| `tp.all_reduce` | **FLOOR** | 929.1 / 870.8 / 878.4 ms |
| `dcp.all_reduce` | **FLOOR** | 250.2 / 265.9 / 258.8 ms |
| `dcp.all_gather` | skew | 291.7 / 135.2 / 146.2 ms |
| `tp.all_gather` | skew | 2.2 / 0.2 / 0.7 ms |

**The two FLOOR terms sum to ~1.18 s.** That is the user's "1.2 s collective
socket", and it is now named: `tp.all_reduce` (~900 ms) plus `dcp.all_reduce`
(~255 ms), both paid by every rank.

Two corrections to how the finding has been carried:

* **"Wait is 2-3x compute" is true of TP1 and TP2 only.** TP0 sits at
  **4.91x**, and for a structural reason worth keeping: TP0 has the *smallest*
  shard, so it finishes compute first (292.6 ms against ~485 ms) and therefore
  waits longest at the barrier. Averaging the ranks hides exactly the
  asymmetry that decides where a lever must go, so the reader reports per rank
  and refuses to pool them.
* **The ratio is batch-dependent, not a constant.** It spans 0.19x to 7.87x
  within a single boot. A single quoted ratio is a summary of one batch shape.

`tp.all_reduce` fires **129 times**, and `2 x 64 layers + 1 = 129` exactly:
one reduce for the attention output and one for the MLP output per layer, plus
one final. That arithmetic identifies the producer as the DENSE per-layer pair
-- which is what the rest of this determination turns on.

### Component -> status

| component | status |
|---|---|
| #252 per-rank compute/wait instrument | **DELIVERED and LIVE BY DEFAULT** (below) |
| #264 prefill-rebalance as the lever | **measured and REFUTED** -- correctly, the dominant term is a floor, not skew |
| #588 token-slice AR pipelining | desk-complete, **coverage measured ZERO**, and structurally unreachable (below) |
| #597 deferred-join at the LayerCommunicator seam | desk-complete, **inert on a dense model** (below) |
| the ~1.18 s floor itself | **STILL OPEN** -- nothing in the tree currently reduces it on this config |

## (b) Do the actuators engage on the serving config? No, and neither is close

### #588 -- two independent blockers

1. `SGLANG_TP_AR_PIPELINE` is `EnvBool(False)` (`environ.py:832`) and **nothing
   in the tree sets it**: grepped `.py/.sh/.yaml/.json/.service` plus
   `os.environ[...] =` assignments; the only hits are the definition, the two
   consumption sites, a test-scoped `.override(True)`, and a manual line in the
   runsheet. No `server_args` field exists.
2. Even exported, the branch is unreachable. `linear.py:2288-2298` requires
   `not is_allocation_symmetric()`, and `is_allocation_symmetric()` is
   `not is_dp_attention_enabled() or is_dp_max_padding()`. On a plain TP=3 boot
   DP attention is off, so the expression is `True` and `not True` is `False`.
   **The pipelined path cannot be taken on this rig's topology at all.**

That also explains window 8's `calls_pipelined == 0` mechanically, rather than
only via the `reduce_results=False` story: the branch additionally requires
`self.reduce_results`, which `qwen3_5` sets False on `o_proj` (`:352`, `:952`).

### #597 -- the issue site is MoE-only, and the model is dense

`issue_deferred_all_reduce` has **exactly one caller in the tree**:
`fused_moe_triton/layer.py:2062`, inside `FusedMoE.forward_impl`, additionally
requiring `self.reduce_results and (moe_tp_size > 1 or moe_ep_size > 1)`.

The serving model never reaches it. `Qwen3.8-27B-INT8-yarn1.5` is **dense**:
`model_type: qwen3_5` (not `qwen3_5_moe`), no `num_experts` /
`moe_intermediate_size`, plain `intermediate_size`. `qwen3_5.py:975` takes the
"Dense MLP for non-MoE variant" branch and `:1323` computes
`is_moe = "moe" in model_type` -> False, so **no `FusedMoE` is ever
constructed**. The three `join_deferred` calls in `communicator.py`
(`:675`, `:873`, `:891`) are therefore permanent no-ops on this config: no
tensor ever carries a handle for them to join.

### The mis-attribution, and why it matters

The #597 runsheet justified that hook site with: *"`qwen3_5` builds
`o_proj`/`out_proj` with `reduce_results=False` and reduces the MoE output
inside the MoE layer."* The second half is not true of the model the 932.2 ms
came from. **Window 8 ran `Qwen3.6-27B-INT8-W8A8`, which is also dense** --
same three checks, plus the decisive arithmetic: 129 calls is `2 x 64 + 1`, the
dense per-layer pattern; an MoE model would not produce it.

So the collective #597 was built to hide is issued by `reduce_results=False`
producers whose reduce belongs to the `LayerCommunicator` -- which is exactly
the case the runsheet's own "Follow-up" section identifies as **"the remaining
unhooked case ... deliberately out of scope here"**. For this fleet that is not
a residue; it is the whole target.

This is not a defect in #597's machinery. The ceiling algebra, the
issue/join counters, the `note_reduce_site` double-reduce guard and the tests
are all sound, and the honest scoping was already written down. What was wrong
was the target attribution, and it is corrected in place in
`RUNSHEET_597_tp_ar_deferred_join.md`.

### #252 -- the one actuator that does engage

`collective_clock.py` reads **no** environment flag, and
`_install_rank_prefill_timer` is called unconditionally from
`SchedulerMetricsReporter.__post_init__` (`metrics_reporter.py:373` ->
`:458-491`), gated only on `device == "cuda"`. The instrument is live on every
CUDA boot.

## (c) What would close this honestly

Because the instrument is already live, closing #588 needs **no new wiring and
no special boot** -- it needs the existing lines read and judged. That is
`bench/588/run_588_socket.py`, and it is not merely mock-smoked: it was run
end to end against the real 147-sample window-8 log, which is where every
number in section (a) comes from.

Writing it found a defect in itself, which is the reason to run rather than
desk-check: the first version pooled every sample from every rank into one
min/max, mixing TIME variation with RANK variation, and so labelled
`tp.all_reduce` -- which every rank pays at 871-935 ms -- as rank-dependent.
Classification is now on per-rank medians with a deliberately loose 10%
tolerance, because the question is "does every rank pay roughly this", not "are
the ranks bit-identical". A 1% tolerance made almost everything "skew" and the
classification useless.

**Window-gated remainder:** one re-run of the same read on today's HEAD, since
every number above is from 2026-08-05 on `Qwen3.6-27B-INT8-W8A8`, while the
fleet now serves `Qwen3.8-27B-INT8-yarn1.5` under a barlink stack that has since
absorbed #603/#622/#632. The read is one command against a boot log; no arm,
no A/B, no special flags.

## Honest rescope

The #588 remainder is **not** delivered by siblings. What is delivered:

* the instrument that measures the socket (live, no flag);
* the refutation of rebalancing as the lever for the floor term (#264, correct);
* two overlap levers that are desk-complete, hermetically tested, and reach
  **nothing** on this configuration.

What remains open is the ~1.18 s floor itself, and the specific unhooked case
is now named at file:line: the `reduce_results=False` producers whose reduce
the `LayerCommunicator` owns (`communicator.py:1204` for the attention half).
Hooking it requires SUPPRESSING the communicator's own reduce, which is the
double-reduce failure mode -- `note_reduce_site` is already planted so that
attempt fails loudly in the suite rather than quietly returning doubled
activations. That is the next slice, and it is a real one, not a re-run.

Two residues that are not this ticket:

* `dcp.all_gather` is genuine SKEW (291.7 / 135.2 / 146.2), and TP0's share
  grows through the boot (5 ms -> 442 ms across one run). That is a balancing
  question, and it is the one place #264's refutation does NOT apply.
* #613's regime gate makes a captured prefill reachable in the concurrent
  short-prompt regime. A captured prefill changes which collectives are
  replayed rather than issued, so any future socket measurement must record
  whether the prefill graph was on -- the two questions are now coupled.
