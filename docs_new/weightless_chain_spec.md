# #143 — chain speculation (topk == 1 MTP/NEXTN) on the weightless-KV lane

Companion to `dcp_triton_spec_verify.md` (#180, the M4 verify split) and to the
Variant-C lane work (#121 → #131 chunked prefill → #133 symmetric decode graph →
#134/#136 tiered KV + prefetch). This is the single authoritative file for the
feature: current state, decisions, rejected approaches, build plan, tests, GPU
recipe. A fresh agent should be able to work from this file alone.

Scope is **chain spec only** — EAGLE / EAGLE3 / NEXTN at
`--speculative-eagle-topk 1`, fixed `k` per boot. Tree verify (`topk > 1`) stays
hard-rejected on the lane and is not touched (#76/#139). Adaptive `k` is out of
scope for V1.

---

## 0. Why this is the decode multiplier

The lane's decode step is already graph-captured and guard-free (#133: 63.5 vs
13.1 tok/s eager, +385%). Every captured step still costs the same four
collectives per full-attention layer regardless of how many tokens it commits.
Chain spec turns one such step into `1..k+1` committed tokens for the *same*
collective count — the multiplier applies directly to the lane's dominant cost.

The corollary is the hard constraint on this task: **the verify step must be
graph-captured too.** A verify that falls back to the eager dispatch pays the
guard's gloo handshake per collective and lands near 13 tok/s; times an accept
length of ~2.5 that is ~33 tok/s, i.e. a *regression* against the 63.5 tok/s
graph decode it replaces. "Spec on the lane, eager verify" is not a shippable
intermediate state, only a debugging one.

---

## 1. Inventory — what the lane already does, code-verified

### 1.1 The asymmetric forward

`model_runner.py:3669` `_forward_raw` intercepts `is_weightless_worker` and runs
`_forward_weightless_worker` (`:3628`) instead of `model.forward`:

| mode | worker behaviour | line |
|---|---|---|
| `IDLE` | early return, **zero** collectives (head's decoder layers skip attention too) | `:3642` |
| `DECODE` | `attn.forward_decode_weightless_worker` per full-attn layer | `:3651` |
| `is_extend()` | `attn.forward_extend_weightless_worker` per full-attn layer | `:3653` |
| anything else | `NotImplementedError` | `:3660` |

`ForwardMode.TARGET_VERIFY.is_extend()` is **True**
(`forward_batch_info.py:106-114`), so a verify batch already routes to the
extend worker dispatch. That is not an accident: `forward_extend_weightless_worker`
carries an explicit verify branch (`flashinfer_backend.py:5476`).

### 1.2 The per-layer collective sequence

Both collectives live in `layers/dcp/comm.py` and are the only two
`guard_dcp_step` call sites in the tree:

| function | line | NCCL ops | guard op-tag |
|---|---|---|---|
| `cp_all_gather_heads_uneven` | `comm.py:147` | 1 × all_gather | `f"ag_heads:{sum(head_counts)}"` (`:172`) |
| `cp_lse_ag_out_ar_mha_uneven` | `comm.py:209` | all_gather(LSE) + all_reduce(out) | `"lse_merge"` (`:226`) |

Per full-attention layer:

```
DECODE  (head :5339 / worker :5378)          fused KV ag, Q ag, lse_merge   = 4 NCCL ops
EXTEND  no prefix                             fused KV ag                    = 1
EXTEND  has prefix / VERIFY (forced)          fused KV ag, Q ag, lse_merge   = 4
```

**Decode and verify therefore have the identical op-tag sequence.** That is the
property the whole feature rests on, and §3.1 states it as the lockstep
invariant.

The prefix decision is `forward_mode`-first on both sides — head
`flashinfer_backend.py:5584`, worker `:5476` — which is the #180 D5 second-door
fix already installed on this path (a verify batch carries no
`extend_prefix_lens`, so anything that falls back to a rank-local
`kv_indices.numel()` test hangs; see `[[rank-lokaler-test-vor-kollektiv]]`).
Until #143 that decision was *duplicated verbatim* in two places, which is
exactly where drift happens; #143 extracts it (§4.9).

### 1.3 Token split / owner rule

The lane has **no private split**. `_dcp_owner_write` is an alias of
`_dcp_write_scatter` (`flashinfer_backend.py:2295`), which uses the weighted
owner rule (`layers/dcp/owner.py:65/99`) when a non-uniform token vector is
installed and plain even-modulo otherwise. What *is* lane-specific is the HEAD
vector `[H, 0, 0, ...]` (`distributed/utils.py:234` `weightless_head_counts`,
installed at `flashinfer_backend.py:673`). Head split ≠ token split.

Consequence for verify: the M4 split's pure decisions in `layers/dcp/owner.py`
(`dcp_verify_paged_lens:221`, `dcp_verify_window_is_disjoint:256`,
`dcp_verify_mask_mode:279`) apply unchanged. There is **no new index math** in
#143 — the paged read is `build_dcp_weighted_kv_indices` over `seq_lens`, the
same expression flashinfer already calls, and the draft block is attended out of
locally-projected k/v.

Under weightless the split is in fact *simpler* than under uneven DCP: the head
projects all q/kv heads and the workers project zero, so the ragged draft→draft
stage is entirely head-local (the worker's `[T,0,D]` contribution makes it a
no-op) and only the paged committed prefix crosses ranks.

### 1.4 Where the draft lives today — solo, and it stays solo

`--speculative-draft-placement solo` (`0f3d217442`) is already the right shape
and is **kept as-is** (#153/#155/#160 line): the draft ModelRunner on the solo
rank holds unsharded draft weights + its own draft KV pool; every other rank
holds a `meta` draft with no pool, no backends, no graphs, and never runs a
draft forward.

- role: `model_runner.py:373` `compute_draft_solo_role`
- shadow draft phase: `eagle_worker_v2.py:1098` `_draft_solo_shadow` — receives
  one broadcast, rebuilds the chain tree locally via `_topk1_chain_meta` (`:849`,
  pure `arange` constants), then runs the *same* verify as split mode
- transport: `_solo_send_draft_tokens` / `_solo_recv_draft_tokens` (`:825`/`:840`),
  **one** `tp_group.broadcast` of `[bs, k]` int64 per round — fixed shape, and
  the entire cross-rank traffic of the draft phase
- shadow early-returns: `:479` (no pool), `:518` (no backends), `:533`/`:914`
  (no graphs), `:877`, `:1037` (no draft forward), `:2047` (no draft prefill),
  `:2113` (no draft extend)

The draft runner is also already exempt from the DCP owner rule
(`flashinfer_backend.py:606`, `:514`, `model_runner_kv_cache_mixin.py:2370`) —
its pool is not token-sharded.

**The premise the solo doc relies on does not hold on the lane, and that is
fine.** Split-mode solo argues no input-side comm is needed because the target's
residual stream is replicated by the per-layer RowParallel all_reduce
(`eagle_worker_v2.py:2039-2048`). On the lane there is no residual stream on a
worker at all — the workers run a meta model. The head is the sole hidden-state
producer *by construction* rather than by all-reduce coincidence, which is a
stronger version of the same property.

### 1.5 The accept sync already exists and is the right primitive

`eagle_utils.py:975-996` — after the accept kernel, rank 0 broadcasts
`(predict, accept_index, num_correct_drafts)` through `capture_safe_tp_broadcast`
(`spec_utils.py:102`, pynccl rather than c10d so it survives graph capture and
co-located ranks). This is the #50 hetero-determinism fix
(`[[hetero-spec-determinismus]]`).

Three properties make it exactly what a weightless worker needs:

1. every shape is a function of `(bs, draft_token_num, max_tree_depth)` —
   rank-uniform, boot-fixed;
2. it is unconditional apart from the rank-uniform `forward_mode.is_idle()`
   early return at `:825`;
3. downstream KV bookkeeping is driven by `accept_lens` → `kv_committed_len`
   (`batch_result_processor.py:619`) → the next `eagle_prepare_for_decode`
   (`eagle_utils.py:1062`). Rejected slots are **never explicitly freed**, they
   are reused. So once a worker receives `accept_lens`, its KV bookkeeping is in
   lockstep for free.

### 1.6 What the graph runner does for verify

`DecodeCudaGraphRunner` serves DECODE **and** TARGET_VERIFY
(`decode_cuda_graph_runner.py:14`). With spec on it sets
`capture_forward_mode = TARGET_VERIFY` and
`num_tokens_per_bs = num_draft_tokens` (`:288-300`) — both derived from
`server_args`, hence rank-uniform. Head and worker capture symmetrically
(#133): the worker records `_capture_one_shape_weightless` (`:1392`) with
metadata prepped out-of-graph and the guard disabled around the whole capture
(`:1026-1050`).

---

## 2. What was missing — the six gaps

Numbered as they are fixed in §4.

| # | site | defect |
|---|---|---|
| G1 | `server_args.py:4044` + `:4679` | two blanket rejects: the lane refuses any `--speculative-algorithm`, and `_handle_dcp_validation` refuses spec under `dcp_size > 1` except for `uneven_weighted_dcp`. The lane forces `dcp_size == tp_size` and is not `uneven_weighted_dcp`, so relaxing only the first is insufficient. |
| G2 | `model_runner.py:459-462` | `is_weightless_head/worker` are **not** gated on `is_draft_worker`, while `flashinfer_backend.py:595` *is*. Enabling a draft model gives a draft ModelRunner on a non-head rank `is_weightless_worker == True` with an attention backend whose `weightless_kv` is `False` → `_forward_weightless_worker` trips `assert self.weightless_kv` (`flashinfer_backend.py:5391`/`:5454`). Latent today only because spec is rejected at boot. |
| G3 | `tp_worker.py:557` vs `:573` vs `:633` | the weightless-worker gloo recv sits **above** the `is_verify` early return, while the head's matching send sits **below** it. Every verify step leaves the workers blocked in `broadcast_pyobj` forever. |
| G4 | `eagle_worker_v2.py:2701-2749` | `verify()` dereferences `logits_output.next_token_logits` (`maybe_detect_nan`, `eagle_sample`, `compute_spec_v2_logprobs`) and commits mamba state. A weightless worker has `logits_output is None` and no GDN state. |
| G5 | `model_runner.py:3704` + `decode_cuda_graph_runner.py:1445` | the guard rule keys on `is_decode_or_idle()`, so TARGET_VERIFY keeps the guard **on** while `is_cuda_graph()` admits it to a captured region (a gloo handshake is not capturable). And the worker's only captured body is the *decode* dispatch — a verify replay would pair a decode body against the head's verify body. |
| G6 | `flashinfer_backend.py:5476` / `:5584` | the verify-first prefix decision is duplicated verbatim; the two copies are the drift surface for the D5 defect's third door. |

Everything else — the verify attention split, the accept broadcast, the solo
draft, the KV slot lifecycle, the draft graph runners' `_wl_block_graph = False`
(`28e9917979`) — was already in place.

---

## 3. Design

### 3.1 The lockstep invariant, stated once

> For a weightless worker, the number, order and op-tags of DCP-group
> collectives in a step are a function of `(forward_mode, has_prefix,
> num_full_attention_layers)` only. `accept_len` is **not** an input.

This holds because:

* every step of a chain-spec round is `TARGET_VERIFY` with
  `T = bs * (k+1)` rows, whatever was accepted last step — `k` is fixed per boot
  and `bs` is scheduler state, not accept state;
* `has_prefix` is forced `True` for verify on both sides, so the branch is taken
  identically regardless of `seq_lens`;
* `accept_len` enters only as *values*: `seq_lens += accept_lens`,
  `kv_committed_len += n`. Values change payload contents, never the schedule;
* accept values reach the workers by broadcast, never by local derivation, so no
  worker can compute a different control flow from a different accept.

The consequence the graph work depends on: **verify's op-tag sequence is
identical to decode's** (`ag_heads:{kv}`, `ag_heads:{q}`, `lse_merge` per layer).
A captured verify graph and a captured decode graph are interchangeable from
NCCL's point of view; only the row count `T` differs, and that is baked into the
capture.

The failure mode this rules out is the one `[[rank-lokaler-test-vor-kollektiv]]`
has now recorded five times: a rank-local quantity (here, "how many tokens did
*I* just commit") deciding whether a group collective is entered. #143 adds no
new such condition, and §5.2 pins that with a test rather than a comment.

### 3.2 Placement: the lane requires solo, it does not re-invent it

The weightless lane and solo draft placement are the same shape with different
names — `weightless_kv_head_rank` plays `_spec_solo_rank`. `model_runner.py:1892-1912`
already routes both through the *same* weight-TP=1 override and the *same*
`_build_weightless_worker_meta_model()`.

So #143 **composes** rather than builds: spec on the lane is admitted only with
`--speculative-draft-placement solo` and
`speculative_draft_solo_rank() == weightless_kv_head_rank`. Split placement is
rejected by name. This buys the entire draft phase — one fixed-shape broadcast
per round, zero draft forwards on the workers — with no new mechanism, and it is
why "SOLO BLEIBT" is a design constraint and not just an existing state.

Rejected alternative: teaching `_forward_weightless_worker` a `DRAFT_EXTEND_V2`
dispatch so the draft could run split across the lane. The workers hold no draft
weights and no draft KV pool; there is nothing for them to dispatch. It would be
a second, weaker copy of solo placement.

### 3.3 The accept decision: receive, do not recompute

The worker cannot run `eagle_sample`'s accept kernel (no logits). It also must
not skip the broadcast (the head would hang). So `eagle_sample` grows a
**receive-only** entry: allocate `predict`, `accept_index`, `num_correct_drafts`
at the rank-uniform shapes and enter the *same* `capture_safe_tp_broadcast`.

This is deliberately not a new channel. The alternative — carrying the accept
result on the lane's existing gloo lockstep channel — would mean two token-sync
mechanisms whose orderings have to be kept consistent, and would give up the
capture-safety `capture_safe_tp_broadcast` was written for.

`src` is generalised from the hard-coded `0` to the lane's head rank. With
`weightless_kv_head_rank != 0` the old constant would broadcast garbage from a
rank that has no logits, silently — a wrong-answer bug, not a hang.

The gloo channel in `tp_worker.py` is *disabled* for verify steps rather than
extended: under spec the authoritative tokens are `predict`, which the accept
broadcast already delivers, and the non-spec path stays byte-identical.

### 3.4 Graphs: capture verify symmetrically, by mode

`_capture_one_shape_weightless` picks its recorded body from
`self.capture_forward_mode` exactly as `_forward_weightless_worker` picks from
`forward_batch.forward_mode` — decode → `forward_decode_weightless_worker`,
target-verify → `forward_extend_weightless_worker`. `capture_forward_mode` is
derived from `server_args`, so head and worker choose identically.

The guard rule becomes "guard OFF whenever a captured region may be entered",
i.e. `is_decode_or_idle() or is_target_verify()`. Both disjuncts are rank-uniform.

Deliberately **not** done: a separate verify graph runner or a second ladder.
With spec on, the target never runs plain DECODE — every generation step is
TARGET_VERIFY — so the existing runner's single mode is the whole story.

### 3.5 V1 exclusions (hard-rejected, not hoped-around)

* `topk > 1` — unchanged, `server_args.py:4029` and `:4659` stay untouched.
* `--speculative-use-rejection-sampling` — solo already rejects it (it needs
  per-step draft probabilities on every verifying rank).
* `FROZEN_KV_MTP` — its draft reads the target KV in place; no single rank holds
  the full target KV under DCP. Already rejected by solo.
* `--weightless-kv-chunked-block-size > 0` and the host-spill tier — the
  streaming block loop is a decode-shaped structure with its own bucketed graph
  ladder (#136a/#136b). Composing it with a verify-shaped capture is a separate
  slice; rejected by name so the combination cannot boot silently.
* adaptive `k` (`--speculative-adaptive`) — a boot-fixed `k` is what makes the
  capture single-shaped. Rejected for V1.
* split draft placement, DFLASH, ngram, standalone, multi-layer eagle.

---

## 4. The change set

| # | file | change |
|---|---|---|
| G2 | `model_executor/model_runner.py` | `is_weightless_head/worker &= not is_draft_worker`, mirroring `flashinfer_backend.py:595`. Every draft-runner site already has an `is_draft_solo_host/shadow` sibling (`:1904`, `:1910`, `:1928`, `:1939`, `:1949`, `:2053`, `:3163`). |
| G5a | `model_executor/model_runner.py` | guard rule `is_decode_or_idle()` → `is_decode_or_idle() or is_target_verify()`. |
| G5b | `model_executor/runner/decode_cuda_graph_runner.py` | `_capture_one_shape_weightless` dispatches by `capture_forward_mode`. |
| G3 | `managers/tp_worker.py` | `is_verify` early return hoisted above the weightless gloo recv, on both sides. |
| G4 | `speculative/eagle_worker_v2.py` | `verify()` takes a weightless-worker path: no nan/inf probe, no logprobs, no mamba commit; `eagle_sample` called in receive mode. |
| G4 | `speculative/eagle_utils.py` | `eagle_sample(..., weightless_recv=False)` receive-only entry; `src` from `spec_accept_broadcast_src()`. |
| G6 | `layers/dcp/lockstep.py` (new) | pure decisions: `weightless_has_prefix`, `weightless_layer_op_tags`, `weightless_step_op_tags`, `chain_spec_verify_rows`, `spec_accept_broadcast_shapes`. |
| G6 | `layers/attention/flashinfer_backend.py` | both prefix call sites go through `weightless_has_prefix`. |
| G1 | `server_args.py` | `_handle_weightless_kv_fastlane`: blanket spec reject → chain-spec admission with named sub-rejects. `_handle_dcp_validation`: weightless disjunct on the CUDA spec reject. |
| G1 | `planner/flags.py` | drop `speculative_algorithm` from the lane's `mutually_exclusive_with`; `_c_dcp_spec` learns the lane. |

---

## 5. Tests

### 5.1 Pure / CPU

`test/registered/unit/distributed/test_weightless_chain_spec.py`

1. **Collective-sequence invariance over accept.** `weightless_step_op_tags`
   for the state reached after `accept = 0` and after `accept = k` is the same
   tuple. Repeated over `bs`, `k`, layer count.
2. **Verify ≡ decode in schedule.** the verify tuple equals the decode tuple,
   which is what makes the symmetric capture legal.
3. **Op-tag source pin.** the tags this module emits are pinned against the
   literal format strings in `layers/dcp/comm.py`, read from source — so a
   rename in `comm.py` fails here instead of drifting.
4. **`weightless_has_prefix` is `forward_mode`-first.** verify → `True` with no
   prefix lengths present at all (the D5 class); extend with no prefix → `False`;
   extend with a prefix → `True`. Plus a source pin that both flashinfer call
   sites call the helper rather than re-deriving.
5. **Verify row count is accept-independent.** `chain_spec_verify_rows(bs, k)`
   `== bs * (k+1)` for every accept.
6. **Accept broadcast shapes are boot-fixed.** `spec_accept_broadcast_shapes`
   matches the tensors `eagle_sample` allocates (`eagle_utils.py:871-875`).
7. **Gate matrix.** in `test_uneven_tp_args.py` style: lane+chain+solo+aligned
   rank passes; lane+spec without solo, with a mismatched solo rank, with
   `topk > 1`, with rejection sampling, with block-decode/host-spill, with
   adaptive — each raises, naming its reason.

### 5.2 Byte-identity claim — RETRACTED, see §8

This section originally read:

> lane **with** chain spec at temperature 0 must be **token-identical** to lane
> **without** spec at temperature 0 [...] a hard equality, not a similarity.

That is false, and Window 5 measured it false with the feature under test
switched off. The claim's premise — "greedy verify accepts exactly the tokens
the target would have emitted" — is true of the accept *rule* and not of the
*logits* the rule reads. §8 replaces it with the oracle that does hold.

The accept-rate instrument in the original text survives unchanged and is the
one part of it that was right: a collapsed accept rate with *correct* tokens
means the verify attention is reading the wrong slots (it still runs). Read
`meta_info.spec_accept_length`, **not** `spec_ema_accept_len`
(`[[spec-acceptance-messfalle]]`).

---

## 6. GPU recipe (main rig: head = 5090, workers = 2× 3080)

Cheapest-first. `--rank-gpu-id` is CUDA order, 5090 = 0
(`[[tp5-emulation-uneven-gguf-bugs]]`), so the lane head rank and the solo draft
rank are both 0.

* **R0 — default path unchanged.** Boot without `--weightless-kv-fastlane`,
  spec on. Must be byte-identical to `integration/r3-probe`.
* **R1 — lane without spec.** The #133/#136 baseline; record graph decode tok/s.
  This is the A arm.
* **R2 — gate matrix on the real binary.** The seven rejections of §5.1.7 must
  raise at boot with their named reasons; the admitted config must boot.
* **R3 — the D5 provocation, first.** 1-token prompt through the lane with spec.
  Must answer, not hang. A hang here is diagnosable; the same hang inside R4 is
  not.
* **R4 — A/B, the anchor.** Same model, same prompts, `temperature 0`, CUDA
  graphs **on** (`[[full-perf-testen]]`):
  - arm A: lane, no spec
  - arm B: lane, `--speculative-algorithm NEXTN --speculative-eagle-topk 1
    --speculative-num-steps 3 --speculative-num-draft-tokens 4
    --speculative-draft-placement solo`
  - **Gate 1** — token identity: B's output token id sequence == A's.
  - **Gate 2** — accept length: `meta_info.spec_accept_length` in a sane band
    (> 1.5 for k=3 on a coherent prompt). A verify reading the wrong slots
    collapses this long before it crashes.
  - **Gate 3** — self-determinism: arm B 3× byte-identical.
  - **Gate 4** — tok/s: B must beat A. If B ≈ A/2, the verify is running eager
    (§0) — check that the worker's captured body is the verify dispatch.
* **R5 — no-hang soak.** 512+ tokens including an EOS-terminated request before
  the length limit (the `B1` lane bug class: EOS desync), plus a bs>1 batch.
* **R6 — rejections stay rejected.** `topk 2` and
  `--weightless-kv-chunked-block-size 2048` with spec must both refuse.

Order: R0 → R1 → R2 → R3 → R4 → R5 → R6.

### R4 result (Window 5, Llama-3.1-8B dense TP=2, EAGLE3 topk 1 solo)

**Gate 4 PASSED.** Full numbers, controls and noise floor in
`INTEGRATION_R3_VALIDATION.md`, section "Window 5".

* Mechanism: `Capture target verify CUDA graph end` on **both** ranks
  (TP0 1.94 s, TP1 1.90 s). The verify is graph-captured symmetrically, so the
  "B ~ A/2 = eager verify" condition of §0 does not arise.
* tok/s B/A per content class: 1.126 / 1.727 / 1.215 / 1.205
  (one_token / code / prose / mixed), against a **boot-to-boot** noise floor of
  2.60 % on raw tok/s. Smallest margin clears it by 4.8x.
* Content-robust restatement, since raw tok/s follows output content: a verify
  round costs **1.22-1.27** plain decode steps and returns **1.38-2.12** tokens.
  `B/A = accept_length / cost_ratio` reproduces the measured ratios.
* Gate 2 passed (accept 2.116 on code). Gate 3 passed on a settled server and
  boot-to-boot; the `011` first-probe pattern is prefix-caching in the harness
  and appears identically in arm A, which runs no speculation.

**Gate 1 as written above is not usable on this vehicle, and this is not a #143
defect.** Two control arms on plain TP=2 (no lane) show that (a) spec alone
already breaks strict token identity at `temperature 0` on the default path, and
(b) the lane alone already changes tokens versus plain TP=2 — the expected
consequence of DCP token-sharding plus LSE merge being a different float
reassociation. Both hold with the feature under test switched off.

Consequence for this document: Gate 1 is **replaced**, not merely restated —
the plain TP=1 solo oracle the lane's other rows use does not carry over either,
because speculation breaks the identity independently of the lane. §8 is the
replacement.

---

## 7. Open points

* R4 Gate 4 is the real risk: the symmetric verify capture is written but has
  never been captured on hardware. If `capture_prepare` cannot build a
  verify-shaped dummy batch on a meta-model worker, the fallback is an eager
  verify — correct but perf-negative, and the task should not ship on it.
* `commit_mamba_states_after_verify` is skipped on the worker because the worker
  never runs GDN. For a hybrid-GDN model the head's GDN state is the only copy;
  this is consistent with the lane (GDN is TP=1 on the head, collective-free) but
  has not been exercised with spec.
* Host-spill / block-decode × spec is rejected, not solved. The streaming
  prefix read (`_wl_blockwise_prefix_return_lse`) is already on the verify path's
  extend dispatch, so the eager combination may in fact work; the graph ladder is
  what does not compose. Worth measuring before building.

---

## 8. The correctness oracle for lane × spec

Window 5 left the feature with a measured throughput win and **no correctness
oracle**: Gate 1 compared against a reference that is not token-identical for
two independent reasons, both present with the feature switched off. This
section is the replacement. It is design plus a GPU recipe; the CPU-side
machinery is built and tested (`tests/determinism/test_spec_oracle.py`,
`test/registered/unit/spec/test_spec_verify_dump.py`), the measured bands are
not yet taken.

### 8.1 Why the obvious candidates do not stand alone

Three candidates were considered.

**(a) Compare against TP=1 solo carrying the SAME spec configuration.** The
appeal is that the spec-induced difference then sits in *both* arms. It does not
cancel — the draft's input is the target hidden state, which the lane changes —
but it becomes common-mode, which is what a reference arm is for. Two things
have to be checked rather than assumed, and both hold:

* *Runnability.* The lane requires `tp_size >= 2`; the reference is the plain
  TP=1 path, where speculation is ordinary. Solo placement at TP=1 is
  degenerate (split and solo both give an unsharded draft), so the reference may
  use either — but which one is used must be recorded, and R7 below A/Bs them.
* *Draft geometry.* The lane admits speculation only with
  `--speculative-draft-placement solo` and an **unsharded** draft on the head
  rank (§3.2). A TP=1 reference's draft is unsharded too. The draft side is
  therefore common-mode by geometry, not by luck — the single most useful
  property of this pairing, because it confines every difference between the
  arms to the target forward.

What (a) does **not** buy is a hard equality. Under speculation a single argmax
flip changes `accept_len`, which changes the shape of every later forward. Two
matched-spec arms fork structurally at the first near-tie. "0 flips over N
tokens" is therefore unattainable *by construction*, and demanding it would
reproduce Window 5's mistake one level up.

**(b) Compare the verified logits rather than the drawn token.** This is the
instrument that makes (a) usable: it turns "they differ" into "they differ by
*this much*, and the fork was at a margin of *this*". It needed a tap, and there
was none — `ModelRunner._determinism_dump_logits` hangs off `ModelRunner.sample`
and early-returns unless `logits.shape[0] == 1`, while a verify's logits are
consumed inside `eagle_sample` and carry `bs * (k+1)` rows. `return_logprob` is
not a substitute: it reports post-preprocessing top-k logprobs for emitted
tokens, while the byte-identity classes are stated on raw pre-preprocessing
logits, and it says nothing about the rejected slots.

**(c) Accept statistics.** The weakest: a correct trajectory with a poor draft
fails it, and a wrong-context verify that keeps accepting passes it. It is kept
as a secondary gate for one reason — it is the only instrument that sees the
lane's characteristic failure, a verify attending over the wrong KV slots, which
still runs and still emits self-consistent tokens and merely stops accepting.

### 8.2 What was built

**(a) + (b) as the primary, (c) as a secondary, plus a replay row.** Neither (a)
nor (b) suffices alone: (a) has trajectories that fork structurally, (b) has no
reference. Together they give a token-aligned, band-quantified, near-tie-gated
verdict on the same evidence standard as the rest of #124.

* `tests/determinism/determinism_harness/spec_trajectory.py` — `VerifyRound` /
  `SpecRun` and the **projection into emitted-token space**. Row `i` of a verify
  matrix is the parent of emitted token `i` of that round (the greedy kernel
  writes `predicts[slot] = target_predict[slot]` at exactly the accepted
  indices). After the projection a spec run *is* a `Trajectory`, so every
  existing primitive applies unchanged and `accept_len` is never an index in the
  oracle — which is what lets two arms with different accept lengths be compared
  at all.
* `ByteIdentityClass.SPEC_NEAR_TIE` — run==run bit-identity plus near-tie-only
  divergence. Same bundle as `SELF_DET_NEAR_TIE`, separate class because the
  reason differs and encoding a spec row under an offload-named class is how doc
  traps start.
* `check_accept_rule_exactness` — reference-free: every emitted token is its
  verify row's argmax. Tests the accept *plumbing* (accept_index, the rank-0
  `predict` broadcast a weightless worker's tokens come from, this harness's own
  projection). It does **not** test the KV context: a verify reading the wrong
  slots emits its own argmax perfectly consistently.
* `check_accept_length_floor` — instrument (c), with its weakness written into
  its docstring so it is not mistaken for a proof.
* `python/sglang/srt/speculative/spec_verify_dump.py` — the tap (b) needed.
  Records the full verify matrix in serving dtype plus the accept result,
  keyed by `accept_index` rather than by an assumed chain layout. Behind the
  existing `--determinism-logits-dump-dir`, default off, never raises into the
  serving path, writes nothing on a weightless worker (which has no logits).

Two matrix rows, both `pending_calibration=True` — they carry **no band**,
because the #124 lesson was that a guessed band can be two orders of magnitude
wrong and a wrong band is worse than no row:

| row | reference | what it reaches |
|---|---|---|
| `weightless_spec_matched` | TP=1 solo, identical spec config | the lane's contribution to the target logits, with the spec shape common-mode |
| `spec_replay_teacher_forced` | non-spec boot, fed the emitted sequence as a prompt | whether the verify attended over the right KV context |

The replay row is the sharpest of the two and the cheapest to run: the reference
does one prefill over the test arm's emitted sequence, so the two arms **cannot
fork** and the comparison stays step-aligned for the whole window. It is still
near-tie-gated — a single prefill over N positions is a third fp path, distinct
from both decode and verify.

### 8.3 The band is not guessable from `weightless_decode`

`weightless_decode`'s measured band is 24.0 (fp16 GGUF 27B, fp32 logits). It
does **not** transfer. The projected rows here come from a verify-shaped
forward, and once the two arms' accept lengths differ the same token index is
scored at different slot positions in the two arms — a further fp difference on
top. The band must be measured on the vehicle, and tightened, never loosened,
from what is measured.

---

## 9. Speculation vs. no speculation at temperature 0 — the #50-family finding

Window 5's control arms showed speculation alone breaking token identity at
`temperature 0` on the **default, non-lane** path. That is worth stating
carefully, because "greedy speculative decoding is lossless" is the usual
selling point and is how this project nearly built a gate.

### 9.1 What is established, from the code

**The accept rule is exact.** `verify_tree_greedy_func` (`eagle_utils.py:587`)
dispatches to `sgl_verify_tree_greedy` or `verify_tree_greedy_kernel_triton`;
both accept draft slot `i` iff `candidates[i] == target_predict[parent]` — plain
integer equality — and write `predicts[parent] = target_predict[parent]`, with
`target_predict = argmax(next_token_logits)` (`eagle_utils.py:1029`). There is no
threshold, tolerance, probability comparison or coin anywhere on this path;
`--speculative-accept-threshold-single` / `-acc` and the uniform coins are
consumed by the **sampling** branch only (`eagle_utils.py:1107-1128`; their
only other mention in the tree is the scheduler's runtime-update allowlist,
not a second consumption site). `temperature 0` takes the greedy branch
(`sampling_params.py:169-172` sets `top_k = 1`), with the caveat that the
predicate is BATCH-wide -- `is_all_greedy = all(top_k <= 1)` over the batch's
requests, so one non-greedy request routes the whole batch into the sampling
branch. Gate runs are single-request, so this does not affect the oracle; it
does mean "temperature 0 implies the greedy verify path" is a statement about
the batch, not about the request.

*Therefore a divergence cannot originate in the accept decision.* It enters
through the target logits, or through something that changes what the logits are
computed over. That is a real narrowing — it rules out an entire class of
suspect — but it is **not** the same as "benign".

**One structural, non-fp divergence source does exist**, and no reassociation
argument covers it: penalties. `eagle_utils.py:989-1009` applies the round's
accumulated penalty vector `repeat_interleave`d across all `k+1` draft positions
(upstream's own comment: "a relaxed version of penalties for speculative
decoding"), so the penalty state does not advance within a round; and
`eagle_prepare_for_decode` calls `cumulate_penalty_output_tokens` once per
round, which takes `req.output_ids[-1]` only — so when a round commits `n`
tokens, `n-1` of them never reach the penalizers at all. With repetition /
presence / frequency penalties set, greedy speculation is therefore **not**
lossless, by construction. Both are inert at default sampling params, so neither
explains Window 5; they are recorded because "greedy spec is lossless" is false
in a way that has nothing to do with floating point.

A grammar mask is **not** in this category, and the obvious guess that it is was
checked and is wrong: `generate_token_bitmask` (`eagle_worker_v2.py:2640`) walks
the draft tree and produces a per-draft-position mask, so the FSM does advance
per position.

### 9.2 What is NOT established

That the remaining difference is *purely* floating-point reassociation. The
expected mechanism — `k+1` positions in one forward means different batch
shapes, a different attention kernel path and a different reduction order — is
consistent with every observation, including the one content class that matched
256/256 (a trajectory with no near-ties has nothing to flip). It is a plausible
explanation, and this whole task exists because a plausible explanation is not a
correctness proof.

**The discriminator, and it is unmeasured.** A reassociation flip happens where
the reference's own margin to the flipped-to token is small; a real verify
defect — wrong KV context, a corrupted tree mask, a desynced accept — flips
tokens the reference was *confident* about, and does so at large margins. That
is exactly `check_near_tie_only_divergence`, and running it needs the verify
logits, which is why §8's tap had to be built before this question could be
answered at all. The margin at the divergence has never been recorded.

Until it is: **no gate anywhere may use spec-on vs spec-off token identity as
evidence** (recorded in `EXCLUDED_CASES['spec_vs_nospec_token_identity']`, with
a `validate_matrix` guard so it cannot be re-added, and in
`FEATURES_VS_UPSTREAM.md`'s status legend so it cannot be counted as
`Cross-checked`).

### 9.3 Upstream relevance

If R8 below shows near-tie-only divergence, the finding is "upstream greedy spec
is lossless up to fp reassociation, and the usual claim should say so" — a
documentation point, worth an upstream issue with the margin data attached. If
it shows confident-margin flips, it is an upstream **defect** in a path the fork
merely inherits, and the fork's own #50 machinery (rank-0 accept broadcast) is
the nearest relative. Either way the vehicle is plain TP=2 with no fork feature
enabled, so the report is reproducible upstream as-is.

---

## 10. GPU recipe for §8 and §9

Cheapest first, and deliberately ordered so a failure is diagnosable where it
happens. `--rank-gpu-id` is CUDA order (`[[tp5-emulation-uneven-gguf-bugs]]`);
resolve the 5090's index at runtime with `/spinning/r3val/gpu_map.py`, never
assume it. Vehicle: the Window-5 one — **Llama-3.1-8B-Instruct dense, TP=2**,
lane head = torch[0], `--rank-gpu-memory-mib 29000,18000`, `--context-length
2048`, `--max-total-tokens 16384`, CUDA graphs ON, `--random-seed 1234` on
**every** boot (sglang randomizes when unset). Spec arm: EAGLE3, `--speculative-eagle-topk 1
--speculative-num-steps 3 --speculative-num-draft-tokens 4
--speculative-draft-placement solo`.

* **R7 — the tap, and the reference-arm assumption.** Boot TP=1 with the spec
  config and `--determinism-logits-dump-dir /tmp/det143/solo`. Check: one
  `rank0_verify*.pt` per verify round; `check_accept_rule_exactness` green on
  the read-back run (`SpecRun.from_dump_records`); `mean_accept_length` matches
  the server's `meta_info.spec_accept_length`. Then repeat with
  `--speculative-draft-placement split`: at TP=1 the two must be **byte-identical**,
  which is what licenses using either as the reference. If they are not, the
  reference arm must use `solo` and the discrepancy is its own finding.
  *Cheapest possible failure point: this needs one GPU and no lane.*
* **R8 — §9, the upstream question, on plain TP=2 with no lane.** Two arms,
  same seed, same prompts, spec on / spec off, both dumping. Read back, project,
  and run `check_near_tie_only_divergence` at the first token divergence.
  Report the **margin**, per content class, alongside the measured per-step
  `max|d|`. This is the number §9.2 says has never been taken; it decides
  §9.3 on its own and does not need the lane at all.
* **R9 — band calibration for `weightless_spec_matched`.** Lane+spec vs TP=1
  solo+spec, 256 tokens, both dumping. Record measured per-token `max|d|`
  (median / p95 / max) over the pre-fork prefix and the fork margin. Fill the
  band in at ~2x the measured max, clear `pending_calibration`, and record the
  measured floor in the row's `notes` exactly as the #124 rows do. Self-det
  rerun on the same boot for the class's run==run obligation.
* **R10 — `spec_replay_teacher_forced`.** Take R9's lane+spec emitted sequence,
  feed it verbatim as the prompt of a single request to a **non-spec** boot of
  the same geometry with `max_new_tokens 1` and `--determinism-logits-dump-dir`,
  and compare that prefill's per-position argmax against the sequence. Same
  band procedure. A confident-margin mismatch here is the wrong-KV-context
  signature and is the one result that would make #143 a defect rather than a
  calibration exercise.
* **R11 — accept-length, content-pinned.** With R9's arms producing the same
  tokens over the compared prefix, re-take the Window-5 observation that the
  lane's accept length runs 8-12 % below the plain path. Window 5 could not
  attribute it because the arms emitted different continuations; over a
  token-aligned prefix it is finally comparable. A residual deficit on identical
  content is a real signal about the draft's input (the lane changes the target
  hidden state the draft consumes), not about the verify.

Volume note: a 128k-vocab fp32 verify matrix at k=3 is ~2 MB per round, so a
256-token arm is a few hundred MB. Bound the gate window and clear
`/tmp/det143` between arms.
