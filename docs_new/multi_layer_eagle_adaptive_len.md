# Adaptive draft length for multi-layer EAGLE (`MultiLayerEagleWorkerV2`)

Task #138. Single authoritative file for this feature: current state (code-referenced),
design decisions, rejected alternatives + why, build plan, test recipes, open points.
A fresh agent should be able to work from this file alone.

Branch: `feat/mleagle-adaptive-len` (worktree `<REPO_PATH>/wt-mleagle-adaptive`,
based on `origin/integration/r3-probe`).

---

## 1. Provenance framing (read first)

The adaptive draft-length CONTROLLER is **upstream sglang**, not a fork feature:
`AdaptiveStepSlot`, `AdaptiveSpeculativeParams`, `resolve_candidate_steps_from_config`,
`--speculative-adaptive`, and `adaptive_unsupported_reason` (including its
"MultiLayerEagleWorkerV2 does not implement adaptive" clause) already exist upstream.

The fork's slice of adaptive is: graph-memory offload for the multi-k capture pools
(#93/#102), the high-accept k=4/5 ladder, the frozen-MTP config wiring, state isolation
+ bs-debounce, and the rank-0-broadcast determinism invariants (#50).

**This task adds a new fork delta:** wiring the (upstream) controller into the
multi-layer EAGLE worker, which upstream explicitly does not support. Do not
re-describe the controller itself as fork work.

---

## 2. Current state — where MTP/EAGLE and multi-layer EAGLE diverge

### 2.1 The adaptive machinery (algorithm-agnostic today)

| Piece | File | Note |
| --- | --- | --- |
| Policy (k ladder, hysteresis, dwell, deadzone) | `python/sglang/srt/speculative/adaptive_spec_params.py:245` `AdaptiveStepSlot` | pure Python, no torch, no worker coupling |
| BS routing + debounce + rung metrics | `adaptive_spec_params.py:607` `AdaptiveSpeculativeParams` | |
| Facade + state swap + graph memory | `adaptive_runtime_state.py:192` `AdaptiveController` | |
| Worker contract | `adaptive_runtime_state.py:177` `AdaptiveSpecWorker` Protocol | `speculative_num_steps`, `build_adaptive_runtime_state()`, `apply_runtime_state()` |
| Per-rung resource bundle | `adaptive_runtime_state.py:22` `SpecRuntimeState` | 8 fields, all SINGULAR backends |
| Accept feedback (the only production feed site) | `managers/scheduler_components/batch_result_processor.py:586` | calls `model_worker.on_verify_complete_cpu(...)` |
| Pre-round activation | `base_spec_worker.py:372` `activate_step_by_batch` | called by the worker itself before drafting |

The controller is already generic. Retrofitting a worker means implementing two
methods and calling two hooks — **not** touching the controller.

### 2.2 EAGLE (`EAGLEWorkerV2`) — the reference wiring

- controller built in `__init__` (`eagle_worker_v2.py:1834`)
- boot state registered + ladder captured in `init_cuda_graphs` (`:1889-1918`)
- `build_adaptive_runtime_state` (`:2379`) / `apply_runtime_state` (`:2440`) /
  `_override_worker_state` (`:2537`)
- `on_verify_complete_cpu` (`:2360`), `activate_step_by_batch` (`:2373`), the latter
  called at `:2074` in the decode branch of `forward_batch_generation`
- k-switch under CUDA graphs: **one fully pre-captured graph set per candidate k,
  captured at boot; never re-captured at runtime.** `_activate` remaps physical pages
  (#93 offload) then swaps python pointers.

### 2.3 Multi-layer EAGLE — the structural divergence

`MultiLayerEagleWorkerV2` subclasses `BaseSpecWorker` (**not** `EAGLEWorkerV2`) and
`MultiLayerEagleDraftWorker` subclasses `EagleDraftWorkerBase` (**not**
`EagleDraftWorker`). Zero shared implementation (deliberate, `#136a`).

Five divergences that matter:

**D1 — k separate draft models, one per chain position.**
`tp_worker.py:405 _init_multi_layer_eagle_model_runners()` builds
`speculative_num_steps` `ModelRunner`s with `draft_model_idx=i`; the weight loader
keeps only MTP layer `i` (`model_loader/loader.py:714 _filter_mtp_weights`,
`models/step3p5_mtp.py:210`). Layer `i` predicts chain position `i+1`.
Consequence: **k is baked into the loaded weights at boot. It can shrink to a prefix
of the loaded layers, never grow past them.** (EAGLE rolls out ONE draft model k
times, so growth there is free.)

**D2 — there is no draft-time model forward at all.**
`MultiLayerEagleDraftWorker.draft_forward` (`multi_layer_eagle_worker_v2.py:329`)
runs **no** model; it only reshapes `spec_info.topk_p / topk_index` into a chain and
calls `build_tree_kernel_efficient`. All k draft forwards happened at the END of the
PREVIOUS round, in `_draft_extend_for_decode` (`:524`), which replays one graph per
step against one shared buffer set and advances the chain with `rotate_input_ids`.

Round shape: `draft()` (tree from carried columns) -> `verify()` -> `_draft_extend_for_decode()`
(produces the NEXT round's columns).

**D3 — `cuda_graph_runner` is always `None`; the only draft graph family is the
composite draft-extend runner.**
`_capture_cuda_graphs` (`:243`) sets `self.cuda_graph_runner = None` and builds
`MultiLayerEagleMultiStepDraftExtendCudaGraphRunner`
(`multi_layer_eagle_draft_extend_cuda_graph_runner.py:438`), which creates
`speculative_num_steps` per-step runners over ONE shared buffer set whose token
window is `num_tokens_per_bs = speculative_num_draft_tokens` (`:172`).
Graph count = `num_steps x len(capture_bs)`; the graph key is batch size only (`:204`).
A different k means a different buffer window AND a different step count -> a wholly
new composite runner. Nothing can be shared across rungs.

**D4 — the per-step graph runner reads `server_args`, not the worker**
(`multi_layer_eagle_draft_extend_cuda_graph_runner.py:147-151`). Any k override must go
through `server_args.override(...)`, i.e. the `_override_worker_state` pattern.

**D5 — draft-extend attention backends are a LIST, one per step**
(`multi_layer_eagle_worker_v2.py:225 init_attention_backend` ->
`draft_extend_attn_backend_list`, and it also rebinds `draft_runner_list[step].attn_backend`).
`SpecRuntimeState` only models a single `draft_extend_attn_backend`, so the M16/#50
isolation guard (`adaptive_runtime_state.py:90 assert_runtime_state_isolation`) would
silently pass while per-step backends 1..k-1 alias across rungs — exactly the
stale-buffer corruption class the guard exists to catch.

### 2.4 Cross-algo bandit (#156) — no shared wiring, keep it that way

The bandit hard-rejects multi-layer EAGLE (`cross_algo_utils.py:715`). Its arms are
`("nextn", k)` / `("dflash", block)` and it attributes results by draft-token stride
(`cross_algo_worker.py:1826`). If it were ever un-gated, a multi-layer rung would be
misattributed as a `nextn` arm. **We change k WITHIN one algorithm; the bandit switches
ALGORITHMS. No double wiring: this task does not touch `cross_algo_*`, and the
`_fail()` at `cross_algo_utils.py:715` stays.**
Shared-but-not-duplicated infrastructure: `SpecRuntimeState`,
`assert_runtime_state_isolation`, `AdaptiveGraphMemoryManager`, `RungMetrics`,
`on_verify_complete_cpu`.

---

## 3. The hard problem: carried draft columns vs. a changed k

Because of D2, the draft columns consumed by round N+1 were produced by round N's
draft-extend at round N's k. The k-switch happens BETWEEN rounds (the controller is fed
from `batch_result_processor`, which runs after `forward_batch_generation` returned),
so the switch never lands mid-round. What it does land on is a **width mismatch**:

```
round N   : draft(k_prev) -> verify(k_prev) -> extend(k_prev)  => carried width k_prev
[scheduler processes result -> controller -> apply_runtime_state(k_new)]
round N+1 : draft(k_new) reads carried columns of width k_prev   <-- MISMATCH
```

`draft_forward` indexes `tree_info[0][:, :, i]` / `tree_info[1][:, i]` for
`i in range(speculative_num_steps)` (`:355-369`), so it needs exactly
`k_new * topk` columns.

**Decision: adapt the carried columns at the start of the round.**

- **downshift** (`k_new < k_prev`): slice the first `k_new * topk` columns. Semantically
  exact — column `i` is MTP layer `i`'s prediction for chain position `i`.
- **upshift** (`k_new > k_prev`): pad by repeating the last column block.

Padding is safe because **speculative decoding is verified by the target model: a bad
draft costs throughput, never correctness.** The padded positions carry a duplicate of
position `k_prev-1`'s token and (under `--speculative-use-rejection-sampling`) a
duplicate of its `q` — and that token WAS drawn from that `q`, so the Leviathan accept
test `coin*q < p` stays exact. One round of degraded draft quality per upshift; from
round N+2 on, the extend has produced genuine `k_new` columns.

Helper: `adapt_draft_columns()` in `multi_layer_eagle_utils.py` — a pure tensor
function, unit-tested on CPU.

### Rejected alternatives (do not re-try)

- **Straddle state `(W_in=k_prev+1, S=k_new)`, captured.** Correct and graph-covered,
  but turns the state set from |C| into up to |C|^2 composites (default |C|=3 -> 9),
  ~2.3x boot capture time and graph memory. Padding gets the same correctness for free.
- **Straddle extend run EAGERLY.** The eager multi-step extend path is
  unvalidated: `prepare_for_draft_extend` only inits metadata on
  `draft_runner_list[0]`'s backend, so steps >= 1 used to run with
  uninitialised/stale attention metadata — hence the standing
  `"can't use cuda graph for draft extend! may have correctness issue!"`
  warning. #184 now plans each rung's backend inside the eager loop, but that
  fix has no GPU evidence yet and the warning stays, so this remains true:
  never route a correctness-critical transition through it. (See open point O1.)
- **Fixed extend window `W = k_max+1` with variable step count.** Needs `k_max` draft
  columns to build the verify tree, i.e. `k_max` MTP forwards — which is exactly the
  cost the ladder exists to avoid. No saving.
- **Always run the extend at `k_max` and only vary the verify width.** Same objection:
  the k MTP forwards ARE the cost; the target verify width is nearly free at bs=1.

---

## 4. Design

### 4.1 What is shared, what is multi-layer specific

**Shared, unchanged:** `AdaptiveStepSlot`, `AdaptiveSpeculativeParams`,
`AdaptiveController`, `AdaptiveGraphMemoryManager`, `RungMetrics`, the
`on_verify_complete_cpu` feed site, the `#50` rank-uniformity invariant.

**Multi-layer specific:**
1. a per-algorithm default config with no step-0 rung (§4.2)
2. the ladder ceiling = number of loaded MTP layers (§4.3)
3. `SpecRuntimeState.draft_extend_attn_backend_list` + isolation walk (§4.4)
4. `build_adaptive_runtime_state` / `apply_runtime_state` / `_override_worker_state`
   for this worker (§4.5)
5. carried-column width adaptation (§3)

### 4.2 Default config: no step 0

Step 0 (nospec) is impossible here: `_draft_extend_for_decode` would loop `range(0)` and
`torch.cat([])` raises; `draft_forward` would build a zero-column tree. Same situation as
frozen-KV MTP. So a dedicated default keyed `MULTI_LAYER_EAGLE`:

```
bs 1  -> candidate_steps [1, 2, 3]
bs 8  -> candidate_steps [1, 2]
bs 32 -> candidate_steps [1]
```

Ceiling 3 matches `_auto_choose_speculative_params` for the multi-layer archs
(`speculative_hook.py:803` returns `(3, 1, 4)` for `MiMoV2*`), so **the default adaptive
config loads exactly as many MTP layers as the default static config** — zero extra
weight VRAM at the default shape.

The algorithm key is derived, not a new `--speculative-algorithm` value:
`adaptive_algorithm_key(server_args)` returns `"MULTI_LAYER_EAGLE"` when
`resolved_view(server_args).enable_multi_layer_eagle` is set, else the raw algorithm
string. Every resolver call site uses it (`speculative_hook`, `server_args.
max_speculative_num_draft_tokens`, the worker), so they cannot disagree.
`--speculative-adaptive-config high-accept` stays available and is also step-0-free.

### 4.3 Ladder ceiling = loaded MTP layers

`_init_multi_layer_eagle_model_runners` loads `max(candidate_steps)` runners when
adaptive is on (else `speculative_num_steps`, unchanged). `--speculative-num-steps` then
means "initial rung", consistent with EAGLE.

Guards (adaptive path only, so the static path is bit-for-bit unchanged):
- every candidate step >= 1 — `MultiLayerEagleWorkerV2._assert_adaptive_supported`,
  called BEFORE the draft worker is constructed (that constructor loads one MTP
  layer's weights per rung, so the rejection must be cheap).
- `max(candidate_steps) <= hf_config.num_nextn_predict_layers` when that attribute
  exists — enforced in `TpModelWorker._num_multi_layer_eagle_draft_runners`, the
  first place the DRAFT model config exists and the last place before the layers are
  loaded. Otherwise the failure surfaces as a cryptic missing-weights error deep in
  the loader.

Two loops that must follow the LOADED count, not the active k (otherwise higher rungs
get runners without embed/head, or skip a weight update):
- `MultiLayerEagleDraftWorker.init_lm_head` (`:222`)
- `update_weights_from_disk` / `update_weights_from_ipc` (`:921`, `:934`)

`EagleDraftInput.ALLOC_LEN_PER_DECODE` (`:151`) is a CLASS-level global; it must be
sized for the MAX rung, not the initial one. Same for the multi-layer term in the draft
input pool (`model_executor/runner/eager_runner.py:78`,
`2 * speculative_num_steps` -> `2 * max_rung`).

### 4.4 Graph strategy: bucketed k only, no runtime re-capture

Identical to EAGLE: one composite draft-extend runner + one target decode graph runner
per candidate k, all captured at boot inside `AdaptiveController.init_states`
(largest-k-first under #93 offload so the max-state reserve is what gets audited).
`_activate` remaps physical pages then swaps pointers. **No re-capture at runtime, ever.**
A rung whose pruned `cuda_graph_bs` is empty captures no graphs (the existing
`disable_cuda_graph=True` branch of `_override_worker_state`).

`SpecRuntimeState` gains `draft_extend_attn_backend_list: list | None = None`
(default None keeps every existing construction site valid), and
`_iter_state_backends` yields its entries as `draft_extend[i]` so the M16/#50 identity
check covers the per-step backends. `draft_attn_backend` / `cuda_graph_runner` are
`None` for this worker (D3).

Graph memory cost per rung k: `k x len(capture_bs)` extend graphs + one target decode
graph set. For the default ladder [1,2,3] that is 6 extend graph sets vs 3 for the
static k=3 path — under `--speculative-adaptive-graph-memory offload` only the active
rung's physical pages are mapped, so resident VRAM stays at max-state size.

### 4.5 Rank-uniform k decision under TP

`[[rank-lokaler-test-vor-kollektiv]]`: a k switch is a GROUP decision — all ranks must
run the same k on the same round or their CUDA graphs desynchronise (NCCL hang).

There is **no broadcast of the decision**. Uniformity comes from the decision being a
pure function of rank-invariant inputs, and this worker satisfies every link:

1. `MultiLayerEagleWorkerV2.verify` calls `eagle_sample` (`:881`), which broadcasts
   `(predict, accept_index, num_correct_drafts)` from `src=0` via
   `capture_safe_tp_broadcast` (`eagle_utils.py:983-996`).
2. `verify` records `speculative_num_draft_tokens` on the result (`:911`), so a delayed
   overlap result is attributed to the rung that produced it.
3. `batch_result_processor.py:563-566` derives the fed list as a pure elementwise
   transform of `result.accept_lens`; `batch_size=len(batch.reqs)` is rank-identical.
4. The policy layer uses a rounds-based dwell and pure-Python window statistics in a
   fixed order -> bit-identical on every rank.

**Audit rule for any future change here:** never let a rank-local quantity (wall clock,
`numel()`, local logits, a per-rank capability check) reach the k decision.
`SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC=1` all-gathers `(swap_ordinal, target_steps)` on
every swap and raises on divergence — use it in the first TP GPU run.

Note (was pre-existing, FIXED in #185): the multi-layer draft picks themselves
(`fast_topk` / `sample_draft_proposal` in `_draft_extend_for_prefill` /
`_draft_extend_for_decode`, plus the in-graph topk returned by
`cgr.replay(step)`) were NOT rank-0-broadcast, unlike
`eagle_worker_v2._broadcast_draft_picks`. On heterogeneous GPUs that was a latent
#50-class divergence in the multi-layer path. It never affected the k decision
(which rides on the broadcast accept counts). `_broadcast_draft_picks` now lives in
`spec_utils.py` and is called once per MTP rung, before the chain rotation carries
the pick into rung i+1's input_ids — see open point O2.

### 4.6 Default: OFF

`--speculative-adaptive` stays `False` by default, exactly like the EAGLE and
frozen-MTP paths. Nothing about the static multi-layer path changes unless the flag is
passed. An unsupported adaptive config is soft-disabled at arg time
(`_maybe_disable_adaptive`) and hard-rejected at worker init
(`_assert_adaptive_supported`, mirroring `FrozenKVMTPWorkerV2`).

---

## 5. Build plan

1. `adaptive_spec_params.py`: `MULTI_LAYER_EAGLE_DEFAULT_ADAPTIVE_CONFIG`,
   `adaptive_algorithm_key()`, `default_adaptive_config_for` extension, drop the
   multi-layer clause from `adaptive_unsupported_reason`.
2. `adaptive_runtime_state.py`: `draft_extend_attn_backend_list` field + isolation walk.
3. `multi_layer_eagle_utils.py`: `adapt_draft_columns()`.
4. `multi_layer_eagle_worker_v2.py`: controller, `_assert_adaptive_supported`,
   `_override_worker_state`, `build_adaptive_runtime_state`, `apply_runtime_state`,
   `on_verify_complete_cpu`, `activate_step_by_batch` + its call site, width adaptation
   in `draft()`, loaded-count loops.
5. `tp_worker.py`: load `max(candidate_steps)` MTP runners under adaptive + guards.
6. Call-site key plumbing: `speculative_hook.py`, `server_args.py`, `eager_runner.py`.
7. Tests: `test/registered/unit/spec/test_multi_layer_eagle_adaptive.py`,
   plus the `test_boot_constructor_integrity.py` entry.

---

## 6. Test recipes

### 6.1 CPU (no GPU needed)

```
cd <REPO_PATH>/wt-mleagle-adaptive
PYTHONPATH=<REPO_PATH>/wt-mleagle-adaptive/python <VENV>/bin/python \
  -m pytest test/registered/unit/spec/test_adaptive_spec_params.py \
            test/registered/unit/spec/test_frozen_mtp_adaptive_init.py \
            test/registered/unit/spec/test_draft_pick_rank_sync.py \
            test/registered/unit/spec/test_adaptive_graph_memory.py \
            test/registered/unit/spec/test_multi_layer_eagle_adaptive.py \
            test/registered/unit/test_boot_constructor_integrity.py -q
```

Baseline before this branch: 118 passed (first four files).

### 6.2 GPU recipe (NOT run in this task — no GPU available)

**Blocker: no multi-layer EAGLE checkpoint is in the local cache.**
`MultiLayerEagleWorkerV2` is selected only by `--speculative-algorithm EAGLE` plus
`enable_multi_layer_eagle`, which the override registry sets for exactly two families:
- `MiMoV2ForCausalLM` / `MiMoV2FlashForCausalLM` (`arg_groups/overrides.py:449`)
- `Step3p5ForCausalLM` / `Step3p7ForConditionalGeneration` (`overrides.py:1021`)

`<MODEL_PATH>/club-3090/models-cache/` contains neither (only Qwen3.5/3.6,
Gemma-4, Llama-3.1 and their EAGLE3/dflash/MTP drafts). The draft checkpoint must
additionally carry >= 3 MTP layers (`num_nextn_predict_layers >= 3`) for the default
ladder — most published MiMo-V2 drafts ship 3.

So the GPU phase needs a download first. Smallest viable: a MiMo-V2 checkpoint with its
bundled MTP head (Step-3.5 is far too large for this rig).

A/B once a checkpoint exists — full perf, CUDA graphs ON, never eager
(`[[full-perf-testen]]`):

```
# A: static baseline
python -m sglang.launch_server \
  --model <mimo-v2> --speculative-algorithm EAGLE \
  --speculative-draft-model-path <mimo-v2-mtp> \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tp 2 --mem-fraction-static 0.82

# B: adaptive (same flags) +
  --speculative-adaptive --speculative-adaptive-graph-memory offload
# optional first-run TP safety net:
  SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC=1
# optional flap stress:
  SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL=25
```

Measure per content class (code / prose / mixed, separately —
`[[benchmark-harness-pflichten]]`): `meta_info.spec_accept_length`,
`spec_verify_ct`/`decode_s`, tok/s, plus the `sglang:spec_num_steps` /
`sglang:spec_num_draft_tokens` / `sglang:spec_ema_accept_len` metrics. Do NOT use
`spec_ema_accept_len` as the accept length (`[[spec-acceptance-messfalle]]`).

Expected signature: code bursts climb to k=3, prose lulls drop to k=1, one
throughput-neutral degraded round per upshift.

The existing e2e shapes to piggyback on:
`test/registered/models_e2e/test_mimo_v2.py:41` (steps 3 / topk 1 / draft 4,
accept-length threshold 2.5) and `test_step3p5_flash_chain_mtp.py:41` (threshold 2.6).

---

## 7. Open points

- **O1 — pre-existing bug, STRUCTURALLY FIXED (#184), GPU EVIDENCE OPEN:** the
  eager (non-graph) multi-step draft-extend ran steps >= 1 with
  uninitialised/stale attention metadata. `prepare_for_draft_extend`
  (`base_spec_worker.py`) only calls `init_forward_metadata` on
  `draft_runner_list[0]`'s backend, then `_draft_extend_for_decode` marks
  metadata ready for all steps, so the forward path will not repair it.
  The eager loop now plans `draft_extend_attn_backend_list[step]` immediately
  before rung `step`'s forward, mirroring the graph path
  (`MultiLayerEagleDraftExtendCudaGraphRunner.replay`), which plans per rung and
  is the path known to work. Guarded to non-NPU, non-idle batches so the two
  documented exceptions keep their behavior.
  What is NOT proven: that the resulting numerics are correct. That needs a GPU
  plus a multi-layer EAGLE checkpoint (MiMo-V2 / Step-3.5, neither in the local
  cache), so the "may have correctness issue" warning DELIBERATELY STAYS in the
  code, extended with a pointer to the fix. CPU tests pin the plan/forward
  interleaving only (`test_multi_layer_eagle_draft_extend_decode.py`).
- **O2 — FIXED (#185):** multi-layer draft picks are now rank-0-broadcast, one
  sync per MTP rung, placed before the chain rotation in all three pick paths
  (prefill loop, decode graph replay, decode eager loop). `_broadcast_draft_picks`
  moved from `eagle_worker_v2.py` to `spec_utils.py` (re-imported under the same
  name in both workers). CPU-tested by a two-rank simulation plus an AST ratchet
  that requires the sync in the pick's own rung loop
  (`test_multi_layer_eagle_draft_extend_decode.py`,
  `test_draft_pick_rank_sync.py`). Not yet exercised on real heterogeneous GPUs.
- **O3 — rejection sampling x upshift padding:** argued exact in §3, but never measured.
  If an accept-rate anomaly shows up right after upshifts under
  `--speculative-use-rejection-sampling`, this is the first suspect.
- **O4 — `war_fastpath_runner`** is inherited from `BaseSpecWorker` and returns the
  TARGET runner, although this worker's last shared-buffer-reading phase is
  `_draft_extend_for_decode` on the draft runners. Conservative today; revisit if
  per-round buffer swapping is ever added.
- **O5 — planner:** `planner/flags.py:2285` still documents adaptive as legal only for
  EAGLE/EAGLE3(+FROZEN_KV_MTP). Its multi-layer archs are not rig-relevant, so the
  comment is updated but the planner's own choices are unchanged.
