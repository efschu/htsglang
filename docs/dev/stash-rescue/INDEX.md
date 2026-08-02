# Stash rescue (task #432)

The repository shares a single `refs/stash` stack across all ~371 worktrees, so
a stash taken in one worktree is visible — and droppable — from every other one.
Seven legacy entries had accumulated there. This directory archives all of them
in git-visible form and the stack was then emptied.

Each entry is preserved twice, redundantly:

1. **A branch at the stash commit itself** — `wip/stash-rescue-<n>-<slug>`. A
   stash commit is an ordinary commit whose tree is the working-tree state, with
   `^1` = the base commit it was taken on and `^2` = the index state. Branching
   the stash commit therefore preserves everything the stash held, including the
   index/worktree distinction, which a flat patch cannot express.
2. **A patch file** — `stash-<n>-<slug>.patch`, the `git stash show -p` output.
   Every patch was verified with `git apply --check` against its recorded base
   commit in a throwaway detached worktree.

To restore an entry, prefer the branch:

    git -C <repo> worktree add /path/to/wt --detach wip/stash-rescue-<n>-<slug>

or apply the patch onto its recorded base:

    git -C <repo> checkout --detach <base-sha>
    git -C <repo> apply docs/dev/stash-rescue/stash-<n>-<slug>.patch

`<n>` is the index the entry held on the stack at rescue time, not a priority.

Restoring is deliberately left to whoever picks a lane back up: several of these
have been partly or wholly superseded by work that landed afterwards, and the
content assessments below say so per entry. None of these branches is a
merge candidate as-is.

## Entries

### 0 — kv-session-offload S2 step 2 / S1b

- Stash commit: `a1a3c94d44ce9e2cfdd1a0b5d1648b145e95cc43`
- Branch: `wip/stash-rescue-0-kv-session-offload-s2`
- Patch: `stash-0-kv-session-offload-s2.patch`
- Original branch: `integration/r2`
- Base commit: `dfc75c008d85b65428a3a976b6c23171fb7ccfd6`
  ("[Perf] kv-session-offload S2/1: plan cache, incremental counts, copy-stream
  double-buffered H2D prefetch")
- Date: 2026-07-23
- Message: "S2 step2 S1b WIP (fable agent, opus handoff 2026-07-23)"
- Files touched: `python/sglang/srt/managers/kv_session_offload.py`
  (+31 / -4)
- Assessment: Smallest entry of the seven. Adds two helpers, `chunk_ceil` and
  `bundle_spillable_sizes`, i.e. the chunk-granular sizing step of the
  session-spill bundling work. Both symbols exist at the current tip, so this
  increment landed in some form; the archived version is the earlier
  hand-off state, not a delta over what is in the tree today.

### 1 — #143 chain-spec lane (paused, draft-graph-capture hang)

- Stash commit: `9e7b243f5b5dbcf474b85063f5e34e0c5e128073`
- Branch: `wip/stash-rescue-1-chain-spec-lane-143`
- Untracked-files branch: `wip/stash-rescue-1-chain-spec-lane-143-untracked`
  (`7afe4df51f2afe16d9f1b5902434ed43770fd466`) — this entry is the only one of
  the seven with a `^3` parent, and its tree is **empty**: the stash was taken
  with untracked capture enabled but nothing untracked was actually present.
  The branch is kept only so the third parent is not silently lost.
- Patch: `stash-1-chain-spec-lane-143.patch`
- Original branch: `integration/r2`
- Base commit: `f756d28cc163dd2cffd0ac8c614d315570282685`
  ("[Guard] #139: broaden the #76 tree-spec guard to the full dcp_tree_mask
  condition")
- Date: 2026-07-20
- Message: "WIP #143 chain-spec lane (paused f756d28cc1) - draft-graph-capture
  hang"
- Files touched: 18 files, +607 / -50 — the largest entry. Concentrated in
  `speculative/eagle_worker_v2.py` (+274) and `model_executor/model_runner.py`
  (+80), plus `server_args.py` (+70), `tests/determinism/determinism_harness/
  matrix.py` (+54), `managers/tp_worker.py`, `mem_cache/memory_pool.py`, the
  decode/draft CUDA-graph runners, `flashinfer_backend.py` and the
  speculative-hook arg group.
- Assessment: A weightless-KV-lane chain-verify path — new `_wl_stub_draft_input`,
  `_wl_build_chain_verify_input`, `_wl_worker_verify` on the EAGLE v2 worker,
  plus a `determinism_dump_accepted_verify_rows` debug hook in the determinism
  harness. Paused mid-debug on a draft-graph-capture hang, so the tree is a
  known-hanging state, not a working one. None of these symbols exists at the
  current tip; this lane did not land under these names. The most substantial
  and least superseded of the seven — treat the hang as the open question if
  the lane is resumed.

### 2 — #108 draft_kv_layout foundation + disjoint --pd-prefill-topology

- Stash commit: `41a0194658ba2a79eafe0b74244620f6b6e5c069`
- Branch: `wip/stash-rescue-2-draft-kv-layout-108`
- Patch: `stash-2-draft-kv-layout-108.patch`
- Original branch: `bugfix/pd-mamba-conv-state-transfer`
- Base commit: `864edb8cb399f31bc727210960f9950145075bf5`
  ("[bugfix] PD mamba conv-state transfer: per-sub-block (q|k|v) head-sharding")
- Date: 2026-07-19
- Message: "WIP preserved: #108 draft_kv_layout foundation + disjoint
  --pd-prefill-topology (server_args/flashinfer/mixin)"
- Files touched: `server_args.py` (+269 / -14),
  `layers/attention/flashinfer_backend.py` (+31),
  `model_executor/model_runner_kv_cache_mixin.py` (+13)
- Assessment: Two unrelated features that happened to share a working tree —
  the `--draft-kv-layout` foundation for #108, and an unrelated
  `--pd-prefill-topology` flag with its `_handle_pd_prefill_topology` handler
  (19 added lines mentioning it). See the landed-check below.

#### Landed-check verdict for entry 2: PARTIALLY LANDED

Checked against tip `d85d964423` by `git log --all --grep='draft_kv_layout'`
and by grepping the three touched files at tip.

- **`--draft-kv-layout` in `server_args.py`: LANDED, and superseded.** The tip
  carries the flag declaration (`server_args.py:1426`) and a considerably
  larger boot gate (`server_args.py:7282` onwards, 19 hits in that file alone).
  Several validation strings are verbatim identical to the stashed version
  (for example "--draft-kv-layout dcp requires the uneven-hybrid weighted "),
  which identifies the tip code as a descendant of this WIP rather than an
  independent reimplementation. The tip version is a strict superset: the
  stashed help text still described 'dcp' as "a lossless memory optimization:
  at temp=0 the verified output is byte-identical to 'replicated'", whereas the
  tip help text documents the measured threshold and the degraded-configuration
  case, and the tip adds gates the stash never had (multi-layer EAGLE,
  cross-algorithm, draft-solo placement, kv-session-offload). `draft_kv_layout`
  is referenced by 10 files at tip, including the `boot_matrix` arms.
- **`flashinfer_backend.py` and `model_runner_kv_cache_mixin.py` hunks: NOT
  LANDED.** Neither file mentions `draft_kv_layout` at tip. These are the two
  hunks that actually token-shard the draft KV pool and pick the
  draft-replicated wrapper — i.e. the runtime half of the foundation, as
  opposed to the flag-and-validation half that did land.
- **`--pd-prefill-topology`: NOT LANDED.** Zero hits for
  `pd_prefill_topology` / `pd-prefill-topology` anywhere at tip, and no commit
  on any ref mentions it apart from the stash commit itself. This part exists
  nowhere else and is the reason entry 2 is worth keeping independently of
  #108.

Per the task definition this verdict is information only; the entry was
archived in full regardless.

### 3 — MoE expert offload, tracked delta

- Stash commit: `833c1cd58586afc3b41ffdb6846dcfc6a2091d1f`
- Branch: `wip/stash-rescue-3-moe-expert-offload`
- Patch: `stash-3-moe-expert-offload.patch`
- Original branch: `feat/moe-expert-offload`
- Base commit: `dcf1d0f95881c2f9880518dc1a0c3bd3294b38e8`
  ("[Feature] TP > num_kv_heads: replicated-KV + token-sharding (generic)")
- Date: 2026-07-19
- Message: "offload-tracked-delta-captured"
- Files touched: `layers/moe/fused_moe_triton/layer.py` (+129),
  `environ.py` (+10). Pure additions, no deletions.
- Assessment: The early expert-offload hook into the fused-MoE Triton layer —
  `_install_expert_offload`, `_run_moe_core_with_offload`, gated by two new
  environment variables `SGLANG_MOE_RESIDENT_EXPERT_FRACTION` and
  `SGLANG_MOE_OFFLOAD_TRACE`. Both env vars exist at tip and
  `SGLANG_MOE_RESIDENT_EXPERT_FRACTION` is referenced across 26 files, so this
  is an ancestor of the shipped expert-offload feature and is superseded by it.
  Archival value only.

### 4 — tree-spec DCP, pre-port WIP

- Stash commit: `cf9747ed091ba4e1fec873f1c77905acad6638bf`
- Branch: `wip/stash-rescue-4-tree-spec-dcp`
- Patch: `stash-4-tree-spec-dcp.patch`
- Original branch: `feat/tree-spec-dcp`
- Base commit: `dcf1d0f95881c2f9880518dc1a0c3bd3294b38e8` (same base as entry 3)
- Date: 2026-07-18
- Message: "tree-spec-wip-preport"
- Files touched: `layers/attention/flashinfer_backend.py` (+188),
  `server_args.py` (+32 / -29)
- Assessment: Builds the ragged tree mask for DCP — `_build_dcp_ragged_tree_mask`
  and `_get_eager_tree_verify_ragged_wrapper`, under
  `SGLANG_UNEVEN_DCP_WEIGHTED`. `_build_dcp_ragged_tree_mask` exists at tip, so
  the mask construction itself landed. Note the surrounding lane is the one
  deliberately guarded off: `topk>1` under uneven DCP is known-wrong and
  perf-negative, and the guard was restored on purpose. This archive is
  therefore reference material for a guarded-off path — do not treat the branch
  as a resumable feature without revisiting that decision first.

### 5 — GGUF m24 vocab ratio WIP

- Stash commit: `71476cc9cf88ca55e3274aeed717f8b2f4147cad`
- Branch: `wip/stash-rescue-5-gguf-vocab-ratio-m24`
- Patch: `stash-5-gguf-vocab-ratio-m24.patch`
- Original branch: `htsglang-gguf`
- Base commit: `cb12f43367778bbc6f007565bf7d7015f262d000`
  ("[Feature] --rank-tp-ratio auto-performance: measured perf-oriented split")
- Date: 2026-07-16
- Message: "m24-vocab-ratio-wip"
- Files touched: 7 files, +379 / -20 — `layers/vocab_parallel_embedding.py`
  (+146), `server_args.py` (+93), `uneven_perf.py` (new, +67),
  `layers/logits_processor.py` (+62), `distributed/utils.py`, `environ.py`,
  `managers/scheduler.py`
- Assessment: Uneven vocabulary sharding for mismatched GPUs — introduces
  `--rank-vocab-ratio` alongside `--rank-mlp-ratio` and `--rank-moe-ratio`,
  padded shard sizing (`uneven_vocab_padded_shard_sizes`), an uneven all-gather
  for logits (`_all_gather_uneven_vocab_logits`), auto-derivation of the ratio
  from the measured perf cache (`derive_vocab_ratio_from_cache`), and
  `SGLANG_UNEVEN_VOCAB_VECTOR`. All three ratio flags and the env var exist at
  tip (`rank_vocab_ratio` in 12 files, `rank_mlp_ratio` in 20), so the feature
  landed; this is its WIP ancestor. Note that one specific vocab split explored
  in this lane was later discarded on its merits — consult the discard register
  before reviving any particular split from this tree.

### 6 — uneven-TP phase 4, package C (incomplete)

- Stash commit: `79a78fa243a95696a96d031aa3ffe25ea86b4cec`
- Branch: `wip/stash-rescue-6-uneven-tp-phase4-paketc`
- Patch: `stash-6-uneven-tp-phase4-paketc.patch`
- Original branch: `feature/uneven-tp`
- Base commit: `c594435e9b98255f4b7f4318df49d6eb2f3e76d0`
  ("[HiCache] Document the exact int8-mamba x hierarchical-cache
  incompatibility")
- Date: 2026-07-13
- Message: "phase4-paketC-unvollendet (Marker/Load-Guards, Agent gestoppt)" —
  package C incomplete, markers and load guards, agent stopped mid-work
- Files touched: 11 files, +202 / -70 — `distributed/utils.py` (+51),
  `model_loader/loader.py` (+32), and per-model edits across `llama.py`,
  `mixtral.py`, `qwen2.py`, `qwen3.py`, `qwen3_5.py`, `qwen3_5_mtp.py`,
  `qwen3_next.py`, `qwen3_vl.py`, `gemma3_causal.py`
- Assessment: The per-model rollout of uneven TP — a shared
  `tp_attention_head_counts` helper in `distributed/utils.py`, a
  `_check_uneven_tp_support(model_class)` load guard in the model loader, and
  the corresponding per-model support markers. Explicitly incomplete: the
  agent was stopped part-way through the model sweep, so coverage across the
  eleven touched models is uneven and unverified. Neither
  `tp_attention_head_counts` nor `_check_uneven_tp_support` exists at tip, so
  this particular marker/guard mechanism did not land under these names, even
  though uneven TP itself shipped by other means. The oldest entry of the
  seven.

## Verification performed before dropping

For every entry, both preservation forms were verified before the stack was
touched:

- branch identity: `git diff <stash-sha> wip/stash-rescue-...` produced zero
  bytes for all eight branches (seven entries plus the one untracked parent);
- patch integrity: every patch is non-empty and `git apply --check` succeeded
  against that entry's recorded base commit, checked out detached in a
  throwaway worktree so no existing branch or worktree was touched.

Nothing was applied or popped onto any existing branch or worktree at any point.
Only after both checks passed for all entries were the stash entries dropped by
SHA, with `git stash list` re-read between drops. The stack is now empty.
