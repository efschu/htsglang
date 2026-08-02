# ANALYSE 442 — evening adoption sweep, six upstream artifacts of 2026-08-02

Desk round, no GPU (`CUDA_VISIBLE_DEVICES=99`). Base: `integration/r3-probe-next2`
tip `546f70f43e`. Branch `feat/adoption-sweep-442`.

Every claim below is anchored at a file:line in the tree it was read from —
`upstream:` for a line of the fetched PR diff, bare paths for this worktree.
Anything not read is marked UNVERIFIED and is not stated as fact.

## Verdicts

| # | Artifact | Verdict | Decisive reason |
|---|---|---|---|
| 1 | PR #33291 DVR (wjie98) | document divergence; harvest one finding | a new opt-in algorithm, not a rollback fix; but it names our #404 open channel outright |
| 2 | PR #33278 Marlin W8A16 (danielafrimi) | skip | it is MXFP8, not MXFP4; and its arch gate splits this rig's TP group |
| 3 | PR #33288 indexer chunking (Leoyzen) | adopt-later, conditional | no card on this rig reaches the path it patches |
| 4 | PR #33276 DSpark NVFP4 (mmangkad) | **adopt now** | same gap in our tree, merged upstream, one line |
| 5 | PR #33287 DSpark DP counts | **adopt now** | same gap in our tree, one line, latent here |
| 6 | Issue #33289 multi-node deadlock | watch + draft comment | diagnosis stuck at zero comments; the reporter asks for exactly our #431 pattern |
| — | PR #33272 closed unmerged | confirmed folded into #33271 | author's own closing comment; defect not reachable here |
| — | PR #33279 Weight Daemon | note only | re-opens the grounds of a BLOCKED planner entry |

## 1 — PR #33291, Decode-Verify-Rollback

`+10421/-77` in 46 files, OPEN, one bot comment. Not a fix to the existing
speculative rollback: it registers three new algorithms
(`DECODE_VERIFY_ROLLBACK`, `..._EAGLE`, `..._DFLASH`) whose purpose is to make
*deterministic* decode fast, and it is opt-in behind
`--enable-deterministic-inference`, `--speculative-eagle-topk 1`, page size 64,
Triton or FA3 target attention, PP=1, no grammar, no dynamic penalties, no
FlashInfer full attention (upstream PR body, "Known Limitations").

### The premise of the briefing is falsified

The briefing said our #404 fix for pool-axis rollback exists in-tree. It does
not. Every #404 commit touches instrumentation only:

* `8f982f042f` — `dual_group_lane.py`, `scripts/dual_group/r404/*`, two test files
* `ae9cbcef04` — the same set plus `test_lane_proposal_perturb_404.py`
* `18b8ad7544` — `scripts/dual_group/r404/boot.sh` only

Nothing under `python/sglang/srt/speculative/` was changed by #404, and
`grep -rn 404 python/sglang/srt/speculative/` is empty. #399 (`883a03d969`)
fixed two real defects — the `_hidden` view aliasing and a stale `_kv_len` —
but both live in `dual_group_lane.py`, i.e. the lane harness, not the
production verify path.

What #404 actually concluded, from its own merge message: committed content was
INVARIANT to a corrupted proposal in the hermetic harness, and the pool coupling
that only real kernels carry — "workspace, GDN per-draft intermediates, graph
buffers, allocator free-list" — was named as exactly what the harness lacks. The
three leak channels are an open question here, not a closed one.

### The three channels against DVR

**Channel 1, KV written at the freed tail.** DVR does not add a rollback-exactness
mechanism. It adds *fences*: `prepare_for_kv_cache_release` waits on
`war_fastpath_read_done_event` and on a new `rollback_done_event` before Radix
donation may reuse the physical slots (upstream `dvr/worker.py`, the
`prepare_for_kv_cache_release` body), with the stated reason that overlap may
have launched one extra DVR round before the prior result finishes. Correctness
still rests on `seq_lens` bounding every later read. Weaker than the briefing
assumed, and unchanged relative to what we already have.

**Channel 2, recurrent per-step intermediates.** This is DVR's headline and it
is a different design from ours, not a stronger version of it.

* Ours (upstream-derived, unmodified here): during TARGET_VERIFY the hybrid
  linear backend writes per-step states into intermediate caches
  (`layers/attention/linear/gdn_backend.py:469-475`), and after acceptance
  `commit_mamba_states_after_verify` (`speculative/spec_utils.py:737`) scatters
  the accepted step back via
  `hybrid_linear_attn_backend.py:1065 update_mamba_state_after_mtp_verify`.
* DVR: keeps a checkpoint at each exact 64-token chunk boundary and replays the
  accepted suffix, with draft state held separately from verified target state
  and Radix publishing only chunk-aligned prefixes (upstream
  `speculative/dvr/state.py`, class docstring and `prepare_for_cache_release`).

DVR's version exists because GDN's chunked kernel advances state at chunk
granularity, and because it must also survive Radix reuse. It is complementary
in idea and incompatible in mechanism: `DVRStateLifecycle.bind_state_adapter`
hard-requires `mamba_track_interval == mamba_cache_chunk_size == chunk_size`.

**Channel 3, workspaces — the finding worth keeping.** DVR's GDN adapter
allocates a per-request `private_conv_state` and copies the boundary conv state
into it before target verify, with this comment verbatim:

> The stock conv kernel is intentionally in-place. Run it on this tiny
> request-owned copy so target verify cannot mutate accepted endpoints.

That is an upstream author stating channel 3's mechanism as a fact about the
stock path. It checks out in our tree:

* `layers/attention/linear/gdn_backend.py:509-521` — the TARGET_VERIFY branch
  calls `causal_conv1d_update(...)` passing `conv_states`, the *persistent*
  cache, alongside `intermediate_conv_window=...`.
* `layers/attention/mamba/causal_conv1d_triton.py:752` —
  `tl.store(conv_state_ptrs_target, new_conv_state, mask)` in STEP 2 is
  unconditional. It is not gated on `SAVE_INTERMEDIATE`, and the spec-decoding
  branch only changes the *source* offset (`:715`,
  `idx_tokens + (1 if IS_SPEC_DECODING else seqlen)`). So the persistent conv
  window is shifted in place during verify, before anything is accepted.
* `layers/attention/hybrid_linear_attn_backend.py:1098` —
  `fused_conv_window_scatter_with_mask` repairs it afterwards from the
  intermediate window at the accepted step.

Our path is therefore mutate-then-repair, DVR's is never-mutate. Single-stream
and synchronous, the repair window is empty and both are correct. The window is
not empty for anything that reads the same pool between the verify forward and
the repair — overlap scheduling, and specifically the dual-group lane, whose
second worker touches the same pool. That is a concrete, cheap, falsifiable
hypothesis for the #404 question, and it is the one thing from #33291 worth
carrying forward.

**Recommendation.** Do not port DVR. Register the channel-3 hypothesis as the
next #404 window's first arm: instrument the persistent `conv_states` bytes
between the verify forward and `update_mamba_state_after_mtp_verify`, and read
them from the lane's second worker. The existing #404 pool-checksum probe
already has surfaces for `conv`; what it lacks is a reader inside that window.

## 2 — PR #33278, dense Marlin W8A16 on SM80/SM90

**Skip.** The format is MXFP8 (E4M3 weights, UE8M0 scales, block 32), not
MXFP4: the diff repacks fp8 and never mentions e2m1, and our own plain-fp8
marlin path already passes `b_q_type=scalar_types.float8_e4m3fn`
(`layers/quantization/marlin_utils_fp8.py:89`). W8A16 means fp8 weights
dequantized in-kernel with activations left in fp16/bf16. No new CUDA: it
reuses `gptq_marlin_gemm` / `gptq_marlin_repack` at `num_bits=8`. Dense linear
only; `Fp8MoEMethod` untouched.

Three reasons beyond "wrong format":

1. **The gate splits this rig.** `80 <= sm < 100` admits sm86 (the 3080s) and
   excludes sm120 (the 5090). One TP group would then run Marlin W8A16 on two
   ranks and the native MXFP8 W8A8 runner on the third — rank-divergent
   numerics inside one all-reduce, the shape `linear.py:217-226` already
   documents from #377.
2. **It lights up a dormant #385 asymmetry.** `Fp8Config` pins
   `weight_block_size = [1, 32]` for mxfp8 (`layers/quantization/fp8.py:381-383`),
   which is asymmetric — precisely what the coupled-dim rule corrects — and
   `linear.py:295` folds the Marlin 128 block only when the exposed block is
   `None`. Against the measured plan recorded at `linear.py:277-280`
   (per-rank intermediate 14888 / 8944 / 8936), all three ranks fail the PR's
   `assert part_size_k % 32 == 0` at weight load. Today this is unreachable
   because mxfp8 needs capability 100 (`fp8.py:315`); the PR's drop to 80 is
   what would make it reachable.
3. State: DRAFT, red CI, no reviews, author states the unit tests were not run.
   It also conflicts with our #192 deterministic-fp8 block inside `fp8.py`.

**Two spin-offs worth doing independently of the PR.**

* Register the eighth alignment sibling: expose `[128, 128]` for `use_mxfp8`
  instead of `[1, 32]`, or make `linear.py:295` fold Marlin's 128 over an
  asymmetric exposed block and not only over `None`. `lcm(32, 128) = 128` is
  what AWQ already imposes for its group 32 (`awq.py:91-95`), so this raises no
  partition tax; it is a missing registration.
* The PR's six format-agnostic Marlin padding helpers would in principle let
  any Marlin path take non-tile-aligned uneven shards without coarsening — a
  real relaxation of the 128-block tax. Separate, larger, needs its own
  falsifier (the tile family is chosen per shape). It should not ride in on an
  MXFP8 PR.

**#398 could not be found in this tree.** No commit, no doc, no `mxfp4` string
in the catalog. MXFP4 here is MoE-only (`mxfp4.py:299-315` returns no dense
linear method on CUDA) and its Marlin path is hard-gated to SM90/SM120
(`mxfp4.py:520-521`), i.e. sm86 is excluded. #33278 touches zero 4-bit code, so
it cannot shrink such a task. Whether #398 is tracked outside this tree is
UNVERIFIED.

## 3 — PR #33288, varlen routing + query-axis chunking

**Adopt-later, conditional.** Complementary to our merged seq-chunk, confirmed
on both sides: ours (#426 item 1, `layers/attention/dsv4/indexer.py:246-260`
and `:329-357`, knob `SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK`) chunks the
**KV/sequence** axis of the torch paged fallback; #33288 chunks the **query**
axis of the deep_gemm varlen path. They are mutually exclusive per call by
construction — #33288 requires `not SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`.

**Why not now: no card on this rig reaches it.** `_DEEPGEMM_MAJORS = (9, 10)`
(`layers/attention/dsv4/indexer_arch.py:45`) excludes both the 5090 (major 12)
and the 3080s (major 8), so every card here takes `BACKEND_TORCH` — the path
#33288 explicitly skips. Porting it now would be desk-written-never-executed
code in exactly the area where that label has already cost us.

**Porting cost onto the #417-restructured indexer**: of 11 hunks, 4 apply with
offset, 3 are mechanical redirects (module path `sglang.jit_kernel.dsv4` vs
`sglang.kernels.ops.attention.dsv4`; `get_schedule()` does not exist here, use
`get_server_args().mem_fraction_static` as `dsa_indexer.py:1054` already does;
`get_is_capture_mode` lives at `model_executor/runner_utils/capture_mode.py:57`),
3 need substantive rework (the PR keys on a `dsa_topk_backend` attribute our
DSV4 mixin does not have, and calls a `topk_transform_512_flashinfer_unfused`
that exists nowhere in `python/`), and 1 is a hard conflict — the PR collapses a
4-branch topk dispatch, ours is 3-branch and is preceded by the 11-line #425
contract comment ("the tail of `logits` beyond `c4_seq_lens` is UNDEFINED"),
which must be relocated or the contract is lost. Band: M, ~1.5–2.5 desk days,
mocked CI only.

**Three defects the port must not import**, all in the routing hunk:

1. `_varlen_arch_ok = is_cuda() and capability[0] >= 9` contradicts our
   `deepgemm_indexer_supported()` — it would route a 5090 into a kernel #417
   established is absent there. Replace with `deepgemm_indexer_supported(device_id)`.
2. It skips only deep-gemm-path / not-CP / not-capture, and does not check
   `is_in_breakable_cuda_graph()`, `is_in_tc_piecewise_cuda_graph()`,
   `_original_forward_mode` or `tbo_parent_token_range`, all of which
   `_can_use_nonpaged_indexer` rejects (`indexer.py:600-621`). Breakable-graph
   prefill is a live path here.
3. No extend-mode guard: outside extend, `extend_start_loc` is `None` and the
   new path silently returns without writing `c4_sparse_page_indices`.

**#395 alignment: no contradiction.** The knob is a byte budget converted to a
row count using local facts, structurally the same shape as the MiB canon, and
the budgeted quantity `query_rows × max_c4_seq_len × 4 B` has no per-rank
geometry term — the C4 indexer is replicated, not head-sharded
(`indexer.py:974-989`: `n_local_heads = n_heads`, both projections
`ReplicatedLinear`). So the #395 failure mode (one number, different bytes per
rank) is structurally absent. Two frictions remain, both pre-existing and both
already carried by our own DSA indexer (`dsa/dsa_indexer.py:1045-1078`): the
only user-facing knob is a *fraction*
(`SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION`, `environ.py:1312`, default 0.2)
whose denominator shifts between free and total memory depending on which
`min()` branch wins — the "fraction of what?" ambiguity the MiB canon exists to
remove — and the budget reads runtime free memory, so the chunk boundary is not
reproducible across boots. The latter is numerically harmless (chunking is
row-independent and `topk_transform_512` is per-row).

**Do first, independently useful:** land the sequence-axis chunk for the
non-SM120 torch paged fallback. `fp8_paged_mqa_logits_torch`
(`indexer.py:120-204`) still gathers the whole context at `:165` and bmm's the
full `[B,S,H]` at `:177`. That is the reference twin the #425 golden pins
compare against, and it closes the same hole on the path this rig actually runs.

## 4 — PR #33276, DSpark loading for hybrid DSV4 NVFP4 (ADOPTED)

Merged upstream `2026-08-02T18:05:10Z`. Cherry-picked here from upstream
`8672eaa9d0` (Mohammad Miadh Angkad), authorship preserved, landing as
`c1ef5b15c8` on this branch.

Gap confirmed here: `model_loader/loader.py:330-333` appended only
`"model.decoder.*"` to the NVFP4 exclusion list, which is how NEXTN exposes the
draft block. DSpark exposes it as `stages.<id>.*` —
`models/deepseek_v4_dspark.py:889` returns `f"stages.{stage_id}.{mapped_rest}"`
for every `mtp.N.*` checkpoint name. A DSpark draft MoE layer therefore fell
through `HybridFp8NvFp4Config.get_quant_method`
(`layers/quantization/modelopt_quant.py:1445`) into
`ModelOptNvFp4FusedMoEMethod` and was read as NVFP4 rather than source MXFP4.

The PR's second commit (`53f3f0ce40`, stale CPU test fixtures) was NOT taken:
our `configs/model_config.py:627` has `think_end_id`, singular, so the existing
fixture is correct here, and `speculative_draft_attention_backend` does not
appear in our copy of `test_model_overrides.py`.

## 5 — PR #33287, DSpark original global token counts (ADOPTED)

Cherry-picked here from upstream `e2654abeb9` (Degeneracy-Evil), authorship
preserved, landing as `579101ccaf` on this branch. Still OPEN upstream.

Gap confirmed: `speculative/dspark_components/dspark_draft.py` builds its draft
`ForwardBatch` by hand and `_fill_dp_moe_sync_metadata` filled only the
speculatively scaled fields, leaving `original_global_num_tokens_cpu` at its
`None` default — where the standard conversion copies it across
(`model_executor/forward_batch_info.py:807`).

**Reachability here, stated honestly.** Upstream crashes with
`TypeError: 'NoneType' object is not iterable` because their
`decode_cuda_graph_runner.can_run_graph` reads
`original_global_num_tokens_cpu`. Ours still reads `global_num_tokens_cpu`
(`model_executor/runner/decode_cuda_graph_runner.py:853-859`), so that crash is
not reachable at this commit. The only reader of the original field here is
`speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py:638`, which
the DSpark path does not enter. The divergence from the standard conversion is
real regardless, the fix is one line and zero-risk, and it becomes load-bearing
the moment that bucket-selection line is refreshed from upstream.

## 6 — Issue #33289, multi-node TP rank-divergence deadlock — WATCH

Thread state as of this sweep: OPEN, **zero comments**, no maintainer triage.
The reporter has already excluded NCCL fabric/config, CUDA graphs, both
attention backends, DSpark ragged scheduling, request-timeout aborts and client
aborts, across five reproductions, and closes by asking:

> If there is a debug flag that makes the two ranks log their batch composition
> per step (`cur_batch_for_debug`-style), a run with that enabled should catch
> the divergent step in the act.

That is our #431 pattern by name. Diagnosis is stuck and the pattern applies, so
a pointer comment is warranted. Draft below, **not posted** — user preview
first, per the public-claims rule.

### Draft comment (NOT POSTED)

> The two stacks put one rank inside a collective and the peer back at
> `_broadcast_reqs_across_ranks`, which places the divergence *before* the
> collective rather than in it — a rank-local decision taken while the peer
> decided otherwise. That family is hard to catch after the fact because
> nothing records what each rank decided, only where it ended up.
>
> One structural note on the artifacts you already have. The scheduler
> watchdog's `dump_info` prints `cur_batch_for_debug` for the rank it runs on,
> and its `is_active` predicate is `scheduler.is_initializing or
> scheduler.cur_batch_for_debug is not None`. On the idle peer
> `cur_batch_for_debug` is `None`, so that watchdog is inactive and never
> dumps — the rank whose decision you actually need is the one that cannot
> report. That may explain why the dumps have not localized it.
>
> What worked for us on the same failure shape (four occurrences, always a
> rank-local condition evaluated before a group collective) was a per-rank
> ordered recorder rather than a snapshot: append `(op, nbytes, path)` at every
> collective dispatch into a bounded in-process ring, plus one JSON line per
> decision written to a per-rank file and flushed immediately. The flush per
> decision is the load-bearing part — a wedged rank never gets to flush a
> buffer, and the entries that matter are the last few before the wedge. A pure
> offline comparator then returns the first index at which two ranks' sequences
> differ, keyed on a total counter so two logs that started at different points
> are not compared as if aligned. Off by default behind one env var; when off
> the hot path costs a single module-global boolean test. The value is that it
> names the first divergent decision instead of the terminal state.
>
> Applied to your capture, the interesting record is not the all-gather that
> hung but whatever the peer did or did not enqueue one step earlier. If a
> maintainer wants it, the same recorder can be scoped to the scheduler loop
> alone — `(step, forward_mode, batch_size, num_tokens)` per rank — which is
> the `cur_batch_for_debug`-style flag you asked for, made ordered.

Every claim in the draft is anchored: the watchdog predicate is
`managers/scheduler_components/invariant_checker.py:502-524`; `cur_batch_for_debug`
is set to `None` at `managers/scheduler.py:4909` and `:5606`; the recorder is
`distributed/device_communicators/barlink_uniformity.py` (`record` at `:150`,
the flushed per-decision append at `:157-179`, the `total` counter rationale at
`:144-147`, `first_divergence` at `:268`, `load_dump_dir` at `:346`, env gates
at `:81-82`, the read-once `_RECORDING` global at `:205`).
The draft deliberately does not name our fork, does not link anything private,
and asserts nothing about their root cause.

## PR #33272 — why it closed unmerged

Confirmed from the timeline, not inferred. Author-closed at
`2026-08-02T18:04:37Z` with: "Closing this in favour of #33271, which now
contains the same commit... The reason is not tidiness — it is that I was wrong
about these being independent." He had split the packed-layer classification
out as a quantization concern, then found #33271's own branch would not load
without it (`AttributeError: 'MergedColumnParallelLinear' object has no
attribute 'weight'` at `deepseek_v2.py:783`) — every number he had reported on
#33271 was produced with this fix applied out of tree. The commit is unchanged
as `9c13c04f` on `sm80-dsv4-sparse-decode`. The briefing's guess was right.

**Not applicable here.** Our tree already immunized all three `.weight`-read
sites by a different route — a `dense_weight_dtype()` helper that returns `None`
for a packed layer: `models/deepseek_v2.py:425`, `:439`, `:856-862`. The
quant-name enumeration still exists at `:844-850` and `:1876-1881`, but it no
longer feeds an unguarded `.weight.dtype`, and the comments at `:851-856` and
`:1882-1887` name GGUF and GPTQ explicitly. The fourth site #33272 guards
(`_q_b_proj_verified_shape`) does not exist here at all. Upstream's
classify-by-built-layer is arguably the cleaner rule; adopting it would be a
style change, not a fix.

## PR #33279 — Weight Daemon, relevance to #305 / #329

`weight_cache/` does not exist in this tree, so #33279 is not an adoption
candidate; it is a note about a rejected design. The PR makes the existing
upstream weight-cache daemon's transport pluggable, adding a `vmm_fd` backend
(VMM allocations passed as file descriptors over `SCM_RIGHTS`) next to the
existing `torch_ipc` one. What matters for us is the daemon's premise, stated
in its own module docstring: it loads weights through the *full* pipeline —
disk → TP shard → quantize — and only then exports them. That directly answers
the objection behind our BLOCKED planner entry `cuda_ipc_weight_import_pd`
(`planner/rejected.py:366-385`), whose recorded reason is that "the postprocess
rewrites the tensors after import, so the shared pages stop being shared". A
daemon that performs the postprocess on its own side before export does not
have that failure mode. For #305 the daemon is a concrete implementation of the
`WARM_GPU` rung (`registry/ledger.py:82`, `registry/rungs.py:148-150`) —
post-quantized weights resident on the card while no engine owns them — which
is the rung whose hot-switch resume time is still an open item from M1. For
#329 it does not help: elastic re-form changes the TP shape, and the daemon's
export is TP-sharded, so a membership change invalidates exactly what it holds.
Revisiting `cuda_ipc_weight_import_pd` on the new grounds is a planner
decision, not something this sweep should flip.

## What was adopted in this branch

Two upstream cherry-picks with authorship preserved, plus two falsifiers:

* `test/registered/unit/model_loader/test_dspark_nvfp4_exclusion_442.py` — 4 tests
* `test/registered/spec/dspark/test_dspark_dp_original_global_num_tokens_442.py` — 3 tests

Can-fail proven by reverting both one-line fixes in place: 3 of the 7 fail
(`test_dspark_draft_experts_are_excluded_from_nvfp4`,
`test_dspark_name_mapping_really_produces_the_stages_prefix`,
`test_original_counts_are_preserved_unscaled`), 7/7 pass with the fixes
restored. The 4 that pass either way are the controls: NEXTN stays excluded,
the target experts stay NVFP4, the scaled counts stay scaled, and the non-DP
path acquires no new write.

Regression check against the base, both under `CUDA_VISIBLE_DEVICES=99`,
`test/registered/unit/model_loader/` + `test/registered/spec/dspark/`:

| tree | result |
|---|---|
| base `546f70f43e` (`/spinning/wt-442-base`) | 28 failed, 384 passed, 11 skipped, 69 subtests passed |
| this branch | 28 failed, 391 passed, 11 skipped, 69 subtests passed |

The failure sets are byte-identical (`diff` of the sorted FAILED/ERROR ids is
empty): six `test_modelopt_loader.py` cases and the `test_dspark_kernel_parity`
subtests, every one of them `RuntimeError: No CUDA GPUs are available` or
`CUDA error: no CUDA-capable device is detected`. Pre-existing and
desk-unrunnable, not caused by this branch. The +7 passes are the two new
files.

## Follow-ups, ordered

1. #404 next window, first arm: read the persistent `conv_states` bytes between
   the TARGET_VERIFY forward and `update_mamba_state_after_mtp_verify`, from the
   lane's second worker. Section 1, channel 3.
2. Register the eighth alignment sibling for mxfp8 (`[128, 128]`, not `[1, 32]`).
   Section 2; independent of #33278.
3. Land the sequence-axis chunk for `fp8_paged_mqa_logits_torch`
   (`indexer.py:120-204`). Section 3; independent of #33288.
4. Decide on the #33289 comment. Section 6.
5. Re-price `cuda_ipc_weight_import_pd` against the daemon premise, or record
   why the entry stands. Section on #33279.
