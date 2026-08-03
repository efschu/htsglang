# #452 — the capturable MoE offload decode path, refuted at boot

Desk analysis of the 2026-08-02 B1–B4 window. Evidence read for this note:
`/spinning/gpu-battery-results/2026-08-02_desync_graph_proof/` (`RESULTS.md`,
`boot_graphs.log`, `decode_{eager1,graphs}.json`,
`expert_stats_{eager,graphs}.tp{0,1,2}ep0.json`) and
`/spinning/wt-desync-port/scripts/dev/443_graph_proof/boot_graphs.sh`. No GPU
was used for this note; every number below is either read out of those
artifacts or derived from them arithmetically, and the derivations are pinned
in `tests/moe_offload/test_capture_replay_contract_452.py`.

Verdict up front: **B4 is structural and the approach does not survive it.
B2's cause is NOT localised, and the window's own inference about it is not
supported by its own artifact.** The gate is restored (#452 commit 1); the
eager offload path at 149.1 ms/token is unaffected and remains the shipped one.

---

## 1. B4 — verified, quantified, structural

### The chain, confirmed line by line

The captured gather does execute inside the graph, so every replay pays it:

| step | file:line |
|---|---|
| capture region wraps the model forward | `model_executor/runner/decode_cuda_graph_runner.py:640` (`with model_capture_mode(): self.capture()`) |
| capture-mode flag set for that region only | `model_executor/runner_utils/capture_mode.py:89`, read at `:58` |
| stream capture around `forward_fn()` | `model_executor/runner_backend/full_cuda_graph_backend.py:126,149` |
| MoE layer takes the ported branch under capture | `layers/moe/fused_moe_triton/layer.py:2143` |
| …and calls the ported step | `layers/moe/fused_moe_triton/layer.py:2185` |
| which issues the gather | `layers/moe/expert_offload.py:2897` |
| the gather itself | `layers/moe/expert_offload.py:2874` — `torch.index_select(pool_dev, 0, src_row, out=self._cap_scratch_dst[attr])` |
| `pool_dev` is a UVA alias of PAGE-LOCKED HOST memory | `layers/moe/expert_offload.py:2249–2270` (`device_view_of_pinned`) |
| issued on the CURRENT stream, by design | `layers/moe/expert_offload.py:2868` (program order is the correctness argument for the routed apply) |

So: **a zero-copy kernel read of host DRAM across PCIe, on the compute stream,
baked into the graph, once per expert-tensor attr per MoE layer per decode
step.** VERIFIED.

### Why it is structural, in bytes

The measurement that settles it comes out of the run's own expert stats
(`residency.h2d_bytes / residency.fetches` per layer, `residency.fetches /
residency.forwards`):

| quantity | value | source |
|---|---|---|
| MoE layers per rank | 43 | `expert_stats_eager.tp1ep0.json` |
| local experts / resident / scratch | 72 / 31 / 6 | boot log, "31/72 experts resident + 6 scratch" |
| bytes per expert row (all attrs) | mean 8.446 MiB (min 7.688, max 12.375) | per-layer `h2d_bytes / fetches` |
| eager fetches per (layer, forward) | 1.503 / 1.133 / 1.035 on TP0 / TP1 / TP2 | `fetches / forwards` |
| **eager H2D per forward** | **0.535 / 0.398 / 0.366 GiB** | product of the two above |
| **captured gather per step** | **2.128 GiB, every step, on every rank** | 43 × C(=6) × 8.446 MiB |

The captured figure has no variance term because there is nothing to vary:
`src_row` is `int32[C]` with a static C, and the slots no routed expert claimed
hold pool row 0 and get gathered anyway
(`prepare_capturable_remap`, `expert_offload.py:2335–2337`). The eager fetch
moves `n_spill` rows because it knows on the host which experts the step
missed — the data dependence a graph forbids.

**5.30x more PCIe bytes on the clock rank** (2.128 / 0.402 GiB, using TP1's
measured 1.133 fetches/layer), against a measured **6.60x** wall clock. The
step is bandwidth-bound, which is why the two numbers are close: eager moves
0.398 GiB in 149.1 ms = **2.67 GiB/s**, at or near this rig's H2D DMA ceiling
for that rank. Predicted captured time at that same rate is ~797 ms against
984.4 ms measured; the ~1.25x residual is the captured path being (i) a
zero-copy gather kernel rather than a copy-engine DMA and (ii) serialised on
the compute stream, where the eager `_fetch` overlaps its copies on a dedicated
stream with fences either side (`expert_offload.py:2731–2738` vs `:2863–2874`).

Pinned at the desk, with executed counterfactuals:
`test_the_captured_gather_moves_the_worst_case_whatever_the_routing`,
`test_the_volume_claim_can_fail`,
`test_the_measured_multiplier_follows_from_the_static_index_length`,
`test_the_captured_path_has_no_copy_stream_to_overlap_with`.

### It does not improve at a larger operating point

`C = worst_case_unique_spill(bs × top_k, E, R) = min(bs × top_k, E − R)`
(`expert_offload.py:2133–2157`). At bs=1 that is 6. At bs=8 it is
`min(48, 41) = 41` — 14.5 GiB per captured step per rank — while the eager
fetch count grows sub-linearly with bs because tokens share experts. The
multiplier gets worse with batch size and with top_k, and improves only as the
hit rate falls, i.e. only as the offload itself gets worse.

### It is not a property of this rig

Per the standing rule that this rig is a lower bound, not a verdict: the
STRUCTURE (a graph moves the worst case; an offload's whole economy is moving
only the miss) holds on any hardware. Only the magnitude scales with the host
link. On a PCIe 5 x16 host link (~55 GiB/s effective) the same step is 39 ms vs
7 ms per token — a 32 ms/token penalty that the graph's launch-overhead saving
(order 1–5 ms/token across 43 layers) cannot repay. The conclusion is not
rig-specific; only its size is.

---

## 2. B2 — not localised, and the window's inference is unsupported

Three suspects were named. All three were tested; none reproduces.

**(a1) Wrong rows from the device-side remap (cumsum rank vs the host
`tolist()` ordering) — REFUTED at the desk, at the boot's own geometry.**
The #443 suite proved bit-identity at E=64/R=32/C=8 with 32 routed slots, i.e.
always slack between `bs × top_k` and C. The boot ran with *none*: `bs × top_k
== C == 6`, the cumsum rank reaching exactly `C−1` on the last spill expert.
`test_the_ported_step_matches_the_eager_one_at_the_boot_geometry` re-runs the
equivalence at E=72/R=31/C=6 for `n_spill` 0…6, static and frozen-hot-set
residency (14 cases): remap bit-identical to the eager path, and every routed
id addresses the row of the expert it routed to, on BOTH paths.

**(a2) The #394 seam's clamp-to-0 rows feeding the pad expert — REFUTED by the
run's own configuration.** The window ran with no cold tier:
`boot_graphs.sh` does `unset SGLANG_MOE_COLD_TIER_SHM` and `unset
SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE`; the boot log prints "MoE cold-expert host
shard (#394) is INERT on this layer"; the expert stats report
`delegated_cold_experts: 0`, `host_shard_reachability: "local-only"`,
`remote_h2d_bytes: 0`. With `self._cold_tier is None`
(`expert_offload.py:2835`) the breach counter is never allocated (`:2842`), so
`breach_counter is None` inside `prepare_capturable_remap` (`:2344`) and the
clamp branch does not execute at all. The seam was not in the picture.

**(a3) A UVA read race against the prefetch stream — REFUTED structurally.**
`self._stream` is referenced only inside `_fetch`
(`expert_offload.py:2731–2738`), the eager path.
`_issue_fetch_capturable` (`:2863–2874`) issues on the current stream and never
touches it. There is no second stream in the captured path for the gather to
race with. (This is also half of B4's residual: the eager path *gets* the
overlap the captured path structurally cannot.)

**(a4) The replay contract itself — tested, and it holds.** This was the named
lesson: the 63 existing tests transcribed the index arithmetic and used a fresh
cache and one step per test, which cannot see a replay bug. New falsifiers
drive the ported step repeatedly over one cache, write the static input buffer
in place as the graph runner does, and poison the scratch region between steps:

* `test_a_step_is_a_pure_function_of_the_current_routing` — a step run after
  eight unrelated steps is bit-identical to the same step run first;
* `test_a_poisoned_scratch_region_is_fully_re_established` — garbage in
  `[R:R+C]` is fully overwritten, because the gather writes all C slots
  (which is precisely the property that costs B4);
* `test_the_poison_check_can_fail` — executed counterfactual: a gather that
  writes only the slot it "needs" leaves the poison and the checker sees it;
* `test_the_gather_writes_through_the_out_view_into_the_resident_buffer`
  plus `test_the_out_view_check_can_fail` — the `out=` view writes in place
  into the storage the GEMM reads. The counterfactual measured two hazards
  worth recording: a mis-shaped `out` that FITS the shared storage is grown in
  place and overwrites its neighbours, and one that does NOT fit causes torch
  to reallocate the shared storage, moving the resident buffer's own
  `data_ptr` — under a captured graph, the address the graph baked. Neither
  fires today (the shape always matches), but the shape equality is
  load-bearing, not decorative.

**So B2's cause is unknown, and the honest reading of the artifact is weaker
than `RESULTS.md`'s.** That document concludes "the replayed path is moving the
wrong data". Its own evidence does not support that:

* Both arms produce fluent, on-topic, technically coherent 128-token
  continuations (`decode_eager1.json`, `decode_graphs.json`). The graph arm's
  text is a well-formed answer to the prompt, not a degraded one.
* At the measured 83.8 % hit rate, roughly 1.1 of 6 routed slots per layer come
  from scratch. A gather landing wrong rows would corrupt ~17 % of expert
  activations in every one of 43 layers. That degrades output; it does not
  produce a different good answer.
* An immediate divergence (character 5, i.e. the second decoded token) is the
  expected signature of an epsilon-level numeric difference under greedy
  decoding, which resolves a near-tie one way or the other and never converges
  again. It is not evidence of magnitude.

**The window lacked its control arm.** Graphs-vs-eager in sglang differ in more
than the MoE fetch — attention-backend metadata, padding to the captured
bucket, kernel selection. Nothing in this run isolates the offload's
contribution. B2 as it stands says only: *the two arms are not bit-equivalent.*
Severity and attribution are open.

Cheapest experiment that would close it, for whoever next holds a window
(none of these need the refuted path un-gated except the last):

1. Boot a SMALL MoE GGUF that fits fully resident (`--rank-moe-resident-fraction
   1.0`, no offload, no capturable path) twice: eager and graphs. If the arms
   still diverge on a greedy prompt, B2 is not about the offload at all and the
   whole question is upstream. This is the control the window skipped, and it
   is the cheapest thing on this list.
2. Two `ARM=eager` boots for the boot-to-boot floor `RESULTS.md` also flags as
   missing.
3. Only then, with `SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE=1`, the B3 poison
   check: perturb one pinned row between capture and replay and confirm the
   output moves.

---

## 3. What a real fix would require, and what it is worth

Effort and yield, both stated; no kill threshold — the ratio is the argument.

### Option 1 — staged async H2D into a graph-stream double buffer

*(the fix shape named in the ticket)*. Replace the zero-copy gather with a
copy-engine DMA on a second stream inside the captured region, fenced with
captured events, writing a double-buffered scratch so the copy for step N+1
overlaps the GEMM of step N.

* **Yield ceiling: ~1.25x.** It can only recover the residual between the
  volume ratio (5.30x) and the wall clock (6.60x). Best case ≈ 787 ms/token,
  still **5.3x slower than the eager path that already works**. It cannot touch
  the 2.128 GiB the graph is obliged to move.
* **Cost:** cross-stream capture with correct fork/join fences, plus double
  buffering: `2 × C × 8.446 MiB = 101 MiB per layer × 43 = 4.25 GiB per rank`.
  The corridor on this rig already breached the 400 MiB floor with the single
  buffer (min free 53/1158/53 MiB in this very window).
* **Verdict: do not build.** Negative yield at multi-GiB VRAM cost.

### Option 2 — CUDA conditional graph nodes (CUDA ≥ 12.4)

The only mechanism that actually reconciles "capturable" with "move only the
miss": a conditional node per scratch slot, predicated on a device-side count.

* **Yield: the whole 5.30x**, i.e. parity with the eager fetch volume, plus
  whatever the graph's launch-overhead saving is worth (still unmeasured, see
  the repricing below).
* **Cost: very large.** torch exposes no conditional-node API; this means
  building the decode graph by hand in a C++ extension and making it co-exist
  with sglang's graph runner and its bucket/replay machinery.
* **Verdict: not now.** Recorded because it is the *only* known mechanism, so
  anyone who revisits this should start here rather than re-deriving Option 1.

### Option 3 — capture the compute, keep the fetch eager

Graph the MoE GEMMs; leave the fetch a host-planned, data-dependent DMA outside
the graph, feeding a static remapped-`topk_ids` buffer the graph reads. Volume
returns to 0.398 GiB/token. The repo already has the mechanism this needs
(`model_executor/runner_backend/breakable_cuda_graph_backend.py`,
`is_in_breakable_cuda_graph`).

* **Yield: unknown, and it is exactly the ticket's unmeasured premise** (below).
  A break per MoE layer reintroduces host work 43 times per step, so what
  survives is the launch-overhead saving on the attention and dense kernels.
* **Cost: moderate**, and it does not remove the `tolist()` sync — the thing
  the ticket set out to remove.
* **Verdict: the only option with a plausibly positive yield, and it should not
  be built before the sizing probe below says there is anything to win.**

> **UPDATE (#462, 2026-08-03): Option 3 IS NOW BUILT, gated OFF.**
> `layers/moe/breakable_offload.py`,
> `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable`. It was built ahead of the sizing
> probe on an explicit build-go, so the verdict above is superseded on
> sequencing only, not on substance: **F2 (the per-layer break cost on DSV4F)
> is still the first measurement of the next window and still gates whether
> the route is worth running.** Nothing about the route is measured — no boot,
> no replay, no ms/verify figure.
>
> Two things this note predicted are confirmed by the build. First, the
> `tolist()` sync is NOT removed, and it turns out to be *irreducible* rather
> than merely awkward: MoE routing is sequential across layers, so no point in
> a step has several layers' routing decisions available to batch into one
> rendezvous, and any scheme that removes it is the in-graph fetch again.
> Second, the volume does return to the eager 0.366–0.535 GiB/token, because
> the fetch never enters the graph.
>
> One thing the note did not anticipate: most of the mechanism already
> existed. `install()` already builds the `[R+C]` slot arena and binds it into
> the layer's parameters, so a graph captured over that layer *already*
> addresses slots — the build added the bridge buffer that publishes which
> expert sits in which slot, a host-side remap, and the `eager_on_graph`
> break. That also removed a cost this note did not count: `_build_lut`'s two
> PAGEABLE host→device copies per layer, host-blocking because `non_blocking`
> is honoured only for pinned memory. Per layer per step, 3 host-blocking
> crossings become 2.
>
> Details, deviations and the BOOT-PENDING list:
> `docs/dev/DESIGN_462_breakable_route.md`. GPU ticket:
> `docs/dev/TICKET_462_f2_and_replay.md`.

### Repricing the ticket's premise

#443 was justified by the `topk_ids.tolist()` sync being the ranked-#2 cause of
the 2.6x decode gap against the club-3090 reference. **That was never measured
in isolation, and this window's numbers make it unlikely.** The eager decode
step moves 0.398 GiB across PCIe in 149.1 ms — 2.67 GiB/s, at or near the
clock rank's H2D ceiling. A step running at ~100 % of its bottleneck link
cannot be dominated by launch overhead; the volume is the floor and the sync's
contribution is bounded by whatever overlap it prevents.

The honest next step is a **sizing probe before any further build**, and it is
cheap: run the eager offload with the per-layer `tolist()` + Python planning
replaced by a cached plan replayed from the previous step (wrong output, but
the right *shape* of work). The delta is an upper bound on everything the whole
capturable programme could ever buy. If that upper bound is small — and the
bandwidth arithmetic says it should be — then the correct action is to close
the line, not to pick an option above.

### Does the capturable-fetch approach survive?

**No, not in this shape.** "Capturable" and "offload" pull against each other
on a bandwidth-bound step: a graph must move the worst case; an offload exists
because moving only the miss is cheaper. They meet only at Option 2, which
torch cannot express today. The mechanism stays in-tree behind the #452
refusal so a future window can measure a candidate against these numbers
without re-deriving them — but the recommendation is to leave it gated and
spend the next window on the B2 control arm (§2, experiment 1) and the sizing
probe above, both of which are cheap and both of which could retire the line
entirely.

**The shipped path is unchanged and unaffected:** eager expert offload,
`--disable-cuda-graph`, 149.1 ms/token measured in this same window.
